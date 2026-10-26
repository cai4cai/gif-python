// SPDX-License-Identifier: BSD-3-Clause
// gzwrite.h - parallel multi-member gzip writer and helpers for writing large buffers to disk.
//
// A file is a small header followed by a data buffer.  The concatenation is split into chunks of `chunk` bytes that are deflated
// independently on OpenMP threads; the gzip members are concatenated into one valid gzip stream (readable by zlib, gzip, nibabel,
// NIfTI readers).  Chunks that are entirely zero are not deflated: the compressed member of n zero bytes is cached per
// (length, level).
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpucommon {

// Page-aligned host buffer whose pages are touched in parallel when it is (re)allocated.
struct HostBuf {
    void *p = nullptr;
    size_t cap = 0;
    void ensure(size_t bytes, int threads = 16);   // grows (contents are not preserved)
    ~HostBuf();
    HostBuf() = default;
    HostBuf(const HostBuf &) = delete;
    HostBuf &operator=(const HostBuf &) = delete;
};

bool all_zero(const uint8_t *p, size_t n);

struct OutFile {
    std::string fn;
    std::vector<uint8_t> head;              // bytes written before the data (e.g. a NIfTI header)
    const uint8_t *data = nullptr;
    size_t nbytes = 0;
    // internal
    std::vector<std::vector<uint8_t>> parts;
    std::vector<const std::vector<uint8_t> *> zparts;
};

// All files are compressed in one parallel loop over chunks, then written (4 files in parallel).  Throws std::runtime_error.
void write_gz_files(std::vector<OutFile> &files, int level, size_t chunk, int nthreads);
// Uncompressed: header, then the data written in parallel stripes.
void write_raw_files(std::vector<OutFile> &files);

}  // namespace gpucommon
