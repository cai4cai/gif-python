// SPDX-License-Identifier: BSD-3-Clause
#include "gzwrite.h"
#include <zlib.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <sys/types.h>

namespace gpucommon {

void HostBuf::ensure(size_t bytes, int threads) {
    if (bytes <= cap) return;
    free(p);
    cap = (bytes + 4095) / 4096 * 4096;
    p = aligned_alloc(4096, cap);
    if (!p) throw std::runtime_error("out of host memory");
#pragma omp parallel for num_threads(threads) schedule(static)
    for (long i = 0; i < (long)(cap / 4096); i++) ((volatile char *)p)[(size_t)i * 4096] = 0;
}
HostBuf::~HostBuf() { free(p); }

bool all_zero(const uint8_t *p, size_t n) {
    size_t k = 0;
    for (; k < n && ((uintptr_t)(p + k) & 7); k++) if (p[k]) return false;
    const uint64_t *w = (const uint64_t *)(p + k);
    size_t nw = (n - k) / 8;
    for (size_t i = 0; i < nw; i++) if (w[i]) return false;
    for (k += nw * 8; k < n; k++) if (p[k]) return false;
    return true;
}

// A gzip member holding `n` zero bytes (cached per length and level)
static const std::vector<uint8_t> &zero_member(size_t n, int level) {
    static std::mutex mu;
    static std::map<std::pair<size_t, int>, std::vector<uint8_t>> cache;
    {
        std::lock_guard<std::mutex> g(mu);
        auto it = cache.find({n, level});
        if (it != cache.end()) return it->second;
    }
    std::vector<uint8_t> z(n, 0), out;
    z_stream s;
    memset(&s, 0, sizeof s);
    if (deflateInit2(&s, level, Z_DEFLATED, 31, 8, Z_DEFAULT_STRATEGY) != Z_OK) throw std::runtime_error("deflateInit2 failed");
    out.resize(deflateBound(&s, n) + 64);
    s.next_out = out.data(); s.avail_out = (uInt)out.size(); s.next_in = z.data(); s.avail_in = (uInt)n;
    if (deflate(&s, Z_FINISH) != Z_STREAM_END) throw std::runtime_error("deflate failed");
    out.resize(s.total_out);
    deflateEnd(&s);
    std::lock_guard<std::mutex> g(mu);
    return cache.emplace(std::make_pair(n, level), std::move(out)).first->second;
}

void write_gz_files(std::vector<OutFile> &files, int level, size_t CH, int nthreads) {
    struct Task { int f; long c; };
    std::vector<Task> tasks;
    for (int f = 0; f < (int)files.size(); f++) {
        size_t total = files[f].head.size() + files[f].nbytes, nch = (total + CH - 1) / CH;
        files[f].parts.assign(nch, std::vector<uint8_t>());
        files[f].zparts.assign(nch, nullptr);
        for (long c = 0; c < (long)nch; c++) tasks.push_back({f, c});
    }
    bool failed = false;
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (long t = 0; t < (long)tasks.size(); t++) {
        try {
            OutFile &F = files[tasks[t].f];
            const long c = tasks[t].c;
            const size_t H = F.head.size();
            const size_t total = H + F.nbytes;
            size_t a = (size_t)c * CH, b = std::min(total, a + CH);
            if (a >= H && all_zero(F.data + (a - H), b - a)) { F.zparts[c] = &zero_member(b - a, level); continue; }
            z_stream z;
            memset(&z, 0, sizeof z);
            if (deflateInit2(&z, level, Z_DEFLATED, 31, 8, Z_DEFAULT_STRATEGY) != Z_OK) throw std::runtime_error("deflateInit2 failed");
            std::vector<uint8_t> &out = F.parts[c];
            out.resize(deflateBound(&z, b - a) + 64);
            z.next_out = out.data(); z.avail_out = (uInt)out.size();
            size_t pos = a;
            if (pos < H) {   // header part
                size_t e = std::min(b, H);
                z.next_in = (Bytef *)(F.head.data() + pos); z.avail_in = (uInt)(e - pos);
                if (deflate(&z, e == b ? Z_FINISH : Z_NO_FLUSH) == Z_STREAM_ERROR) throw std::runtime_error("deflate failed");
                pos = e;
            }
            if (pos < b) {
                z.next_in = (Bytef *)(F.data + (pos - H)); z.avail_in = (uInt)(b - pos);
                int r = deflate(&z, Z_FINISH);
                if (r != Z_STREAM_END) throw std::runtime_error("deflate did not finish");
            }
            out.resize(z.total_out);
            deflateEnd(&z);
        } catch (...) {
#pragma omp atomic write
            failed = true;
        }
    }
    if (failed) throw std::runtime_error("gzip compression failed");
#pragma omp parallel for schedule(dynamic, 1) num_threads(4)
    for (int f = 0; f < (int)files.size(); f++) {
        FILE *fp = fopen(files[f].fn.c_str(), "wb");
        if (!fp) { failed = true; continue; }
        setvbuf(fp, nullptr, _IOFBF, 1 << 20);
        for (size_t c = 0; c < files[f].parts.size(); c++) {
            const std::vector<uint8_t> &v = files[f].zparts[c] ? *files[f].zparts[c] : files[f].parts[c];
            fwrite(v.data(), 1, v.size(), fp);
        }
        fclose(fp);
    }
    if (failed) throw std::runtime_error("cannot write output file");
}

void write_raw_files(std::vector<OutFile> &files) {
    for (auto &F : files) {
        FILE *fp = fopen(F.fn.c_str(), "wb");
        if (!fp) throw std::runtime_error("cannot write " + F.fn);
        fwrite(F.head.data(), 1, F.head.size(), fp);
        fclose(fp);
    }
    struct Task { int f; size_t a, b; };
    std::vector<Task> tasks;
    const size_t S = 32u << 20;
    for (int f = 0; f < (int)files.size(); f++)
        for (size_t a = 0; a < files[f].nbytes; a += S) tasks.push_back({f, a, std::min(files[f].nbytes, a + S)});
    bool failed = false;
#pragma omp parallel for schedule(dynamic, 1) num_threads(8)
    for (long t = 0; t < (long)tasks.size(); t++) {
        OutFile &F = files[tasks[t].f];
        FILE *fp = fopen(F.fn.c_str(), "r+b");
        if (!fp) { failed = true; continue; }
        fseeko(fp, (off_t)F.head.size() + (off_t)tasks[t].a, SEEK_SET);
        fwrite(F.data + tasks[t].a, 1, tasks[t].b - tasks[t].a, fp);
        fclose(fp);
    }
    if (failed) throw std::runtime_error("cannot open output file");
}

}  // namespace gpucommon
