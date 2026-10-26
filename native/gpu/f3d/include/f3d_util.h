// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                               \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(1);                                                                           \
        }                                                                                      \
    } while (0)

namespace f3d {
// device memory pool: freed blocks are kept and reused (cudaMalloc/cudaFree of GB sized buffers cost 10-100 ms each); F3D_NOPOOL=1 disables
void *pool_malloc(size_t bytes);
void pool_free(void *p);
void pool_trim();   // release all cached blocks
template <class T> T *dalloc(size_t n)
{
    return (T *)pool_malloc(n * sizeof(T));
}
template <class T> T *dalloc_zero(size_t n)
{
    T *p = dalloc<T>(n);
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}
template <class T> std::vector<T> d2h(const T *d, size_t n)
{
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
template <class T> T *h2d(const T *h, size_t n)
{
    T *d = dalloc<T>(n);
    CK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
inline int cdiv(size_t a, size_t b) { return (int)((a + b - 1) / b); }
}  // namespace f3d
