// SPDX-License-Identifier: BSD-3-Clause
// Round trip test of gzwrite: writes buffers with zero chunks and random chunks (gz and raw), reads them back with zlib gzread.
// usage: test_gzwrite [dir=/tmp]
#include "gzwrite.h"
#include <zlib.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace gpucommon;
int main(int argc, char **argv) {
    std::string dir = argc > 1 ? argv[1] : "/tmp";
    std::mt19937 rng(5);
    int bad = 0;
    for (int lev : {1, 6}) for (size_t CH : {(size_t)4096, (size_t)1 << 16, (size_t)1 << 20}) {
        const size_t n = 3 * (1u << 20) + 123;
        HostBuf hb; hb.ensure(n);
        uint8_t *d = (uint8_t *)hb.p;
        memset(d, 0, n);
        for (size_t i = (1u << 20); i < (1u << 20) + 300000; i++) d[i] = (uint8_t)(rng() & 0xff);      // random block
        for (size_t i = 2 * (1u << 20); i < 2 * (1u << 20) + 50000; i++) d[i] = (uint8_t)(i % 7);      // compressible block
        std::vector<OutFile> files(2);
        for (int k = 0; k < 2; k++) {
            files[k].fn = dir + "/gzw_test_" + std::to_string(k) + ".bin.gz";
            files[k].head.assign(k == 0 ? 352 : 5, (uint8_t)(0x40 + k));
            files[k].data = d + k * 1000; files[k].nbytes = n - k * 1000;
        }
        write_gz_files(files, lev, CH, 4);
        for (auto &F : files) {
            gzFile g = gzopen(F.fn.c_str(), "rb");
            std::vector<uint8_t> back(F.head.size() + F.nbytes + 16);
            int got = gzread(g, back.data(), (unsigned)back.size());
            gzclose(g);
            bool ok = got == (int)(F.head.size() + F.nbytes) && !memcmp(back.data(), F.head.data(), F.head.size()) &&
                      !memcmp(back.data() + F.head.size(), F.data, F.nbytes);
            if (!ok) { bad++; printf("FAIL gz level %d chunk %zu %s (read %d)\n", lev, CH, F.fn.c_str(), got); }
            remove(F.fn.c_str());
        }
        for (auto &F : files) F.fn = F.fn.substr(0, F.fn.size() - 3);
        write_raw_files(files);
        for (auto &F : files) {
            FILE *f = fopen(F.fn.c_str(), "rb");
            std::vector<uint8_t> back(F.head.size() + F.nbytes);
            size_t got = fread(back.data(), 1, back.size(), f);
            fclose(f);
            bool ok = got == back.size() && !memcmp(back.data(), F.head.data(), F.head.size()) && !memcmp(back.data() + F.head.size(), F.data, F.nbytes);
            if (!ok) { bad++; printf("FAIL raw %s\n", F.fn.c_str()); }
            remove(F.fn.c_str());
        }
    }
    printf(bad ? "FAIL (%d)\n" : "PASS gzwrite round trip\n", bad);
    return bad != 0;
}
