# gpu/common - host utilities

`gzwrite.h/.cpp`: parallel multi-member gzip writer (chunks deflated independently on OpenMP threads and concatenated into one valid
gzip stream; all-zero chunks are taken from a cache), parallel raw writer and a page-aligned, parallel pre-faulted host buffer.
Files are described by a header byte string plus a data pointer.  `make test` round-trips random/zero/compressible buffers through
zlib's `gzread`.  Requires zlib and OpenMP.
