// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cmath>
namespace f3d {
__device__ __forceinline__ bool valid_vox(const float *ref, const int *mask, float w, size_t i)
{
    float r = ref[i];
    return (mask == nullptr || mask[i] >= 0) && (r == r) && (w == w);
}
struct Stats { float mr, mw, sr, sw, corr; };
// o[0..5]: smoothed channels (m, r, r^2, w, w^2, r*w); NiftyReg normalises by the smoothed density
__device__ __forceinline__ Stats stats_from(const float *o)
{
    float d = o[0];
    Stats s;
    s.mr = o[1] / d;
    float qr = o[2] / d;
    s.mw = o[3] / d;
    float qw = o[4] / d;
    s.corr = o[5] / d;
    s.sr = sqrtf(qr - s.mr * s.mr);
    s.sw = sqrtf(qw - s.mw * s.mw);
    if (s.sr < 1.e-06f) s.sr = 0.f;
    if (s.sw < 1.e-06f) s.sw = 0.f;
    return s;
}
__device__ __forceinline__ double block_sum_d(double v, double *sh)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    double r = 0.0;
    if (threadIdx.x < 32) {
        r = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0;
        for (int o = 16; o > 0; o >>= 1) r += __shfl_down_sync(0xffffffffu, r, o);
        if (threadIdx.x == 0) sh[0] = r;
    }
    __syncthreads();
    r = sh[0];
    __syncthreads();
    return r;
}
}  // namespace f3d
