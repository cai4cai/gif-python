// SPDX-License-Identifier: BSD-3-Clause
// LNCC (NiftyReg reg_lncc, Gaussian kernel) value and voxel-based gradient, batched over templates.
#include "f3d_internal.h"
#include "f3d_lncc_common.cuh"
#include <cmath>

namespace f3d {

__device__ __forceinline__ double warp_sum(double v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
// block reduction of two doubles; result valid in thread 0
__device__ __forceinline__ void block_sum2(double &a, double &b)
{
    __shared__ double sa[32], sb[32];
    a = warp_sum(a); b = warp_sum(b);
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) { sa[w] = a; sb[w] = b; }
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    a = (threadIdx.x < nw) ? sa[threadIdx.x] : 0.0;
    b = (threadIdx.x < nw) ? sb[threadIdx.x] : 0.0;
    if (w == 0) { a = warp_sum(a); b = warp_sum(b); }
}

void lncc_init(LnccCtx &c, const int dims[3], const float pixdim[3], float sigma_mm, const float *d_ref, const int *d_mask, int B, bool exact)
{
    for (int i = 0; i < 3; i++) c.dims[i] = dims[i];
    c.N = (size_t)dims[0] * dims[1] * dims[2];
    c.B = B;
    for (int a = 0; a < 3; a++) c.gt[a] = make_taps(sigma_mm, pixdim[a], 0);
    c.ref = d_ref; c.mask = d_mask;
    c.bufA = dalloc<float>(6 * c.N * B);
    c.bufB = dalloc<float>(6 * c.N * B);
    c.bufC = dalloc<float>(6 * c.N * B);
    c.nblk = cdiv(c.N, 256);
    {
        int tiles = cdiv(dims[0], 16) * cdiv(dims[1], 16);
        c.nblk_s = tiles;
        int mx = c.nblk > tiles ? c.nblk : tiles;
        c.partial = dalloc<double>((size_t)B * mx * 2);
        c.partStride = mx;
    }
    c.exact = exact && getenv("F3D_FAST") == nullptr;
    c.lfver = getenv("F3D_LF1") ? 1 : 2;
    c.fused = lncc_stream_supported(c) && getenv("F3D_UNFUSED") == nullptr;
    c.acc = dalloc_zero<double>((size_t)B * 4);
    lncc_build_ref_tables(c);
    if (getenv("F3D_VERBOSE")) fprintf(stderr, "lncc_init dims %d %d %d radius %d %d %d fused %d\n", dims[0], dims[1], dims[2], c.gt[0].radius, c.gt[1].radius, c.gt[2].radius, (int)c.fused);
}
// smoothed reference channels (m, r, r^2) of an all-valid warped image: the arithmetic of the generic exact convolution
__global__ void ref_prep_k(const float *ref, const int *mask, float *A, size_t N)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float r = ref[i];
    bool m = (mask == nullptr || mask[i] >= 0) && (r == r);
    A[i] = m ? 1.f : 0.f;
    A[N + i] = m ? r : 0.f;
    A[2 * N + i] = m ? r * r : 0.f;
}
void lncc_build_ref_tables(LnccCtx &c)
{
    c.Px = c.Pxy = c.Pxyz = nullptr;
    if (!getenv("F3D_LNCC_SKIP")) return;   // experimental (not faster than the plain kernel)
    if (!c.exact || !c.fused || c.gt[0].radius != 8 || c.dims[0] <= 1 || c.dims[1] <= 1 || c.dims[2] <= 1) return;
    float *A = dalloc<float>(3 * c.N);
    c.Px = dalloc<float>(3 * c.N); c.Pxy = dalloc<float>(3 * c.N); c.Pxyz = dalloc<float>(3 * c.N);
    ref_prep_k<<<cdiv(c.N, 256), 256>>>(c.ref, c.mask, A, c.N);
    conv_axis(A, c.Px, 3, 1, 3 * c.N, c.dims, 0, c.gt[0], 0, true);
    conv_axis(c.Px, c.Pxy, 3, 1, 3 * c.N, c.dims, 1, c.gt[1], 0, true);
    conv_axis(c.Pxy, c.Pxyz, 3, 1, 3 * c.N, c.dims, 2, c.gt[2], 0, true);
    CK(cudaDeviceSynchronize());
    pool_free(A);
}
void lncc_free(LnccCtx &c)
{
    if (c.Px) { pool_free(c.Px); pool_free(c.Pxy); pool_free(c.Pxyz); c.Px = c.Pxy = c.Pxyz = nullptr; }
    for (int a = 0; a < 3; a++) free_taps(c.gt[a]);
    pool_free(c.bufA); pool_free(c.bufB); pool_free(c.bufC); pool_free(c.partial); pool_free(c.acc);
}

// channels: 0 m, 1 r, 2 r^2, 3 w, 4 w^2, 5 r*w   (all zeroed where m==0)
__global__ void lncc_prep_k(const float *ref, const int *mask, const float *warped, float *A, size_t N, const int *act)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int b = blockIdx.y;
    if (act && !act[b]) return;
    float w = warped[(size_t)b * N + i];
    float r = ref[i];
    bool m = valid_vox(ref, mask, w, i);
    float *a = A + (size_t)b * 6 * N;
    a[i] = m ? 1.f : 0.f;
    a[N + i] = m ? r : 0.f;
    a[2 * N + i] = m ? r * r : 0.f;
    a[3 * N + i] = m ? w : 0.f;
    a[4 * N + i] = m ? w * w : 0.f;
    a[5 * N + i] = m ? r * w : 0.f;
}

__device__ __forceinline__ Stats get_stats(const float *B6, size_t N, size_t i)
{
    float o[6] = {B6[i], B6[N + i], B6[2 * N + i], B6[3 * N + i], B6[4 * N + i], B6[5 * N + i]};
    return stats_from(o);
}

__global__ void lncc_value_k(const float *ref, const int *mask, const float *warped, const float *Bf, size_t N,
                             double *partial, int nblk, const int *act)
{
    int b = blockIdx.y;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0, cnt = 0.0;
    if (!(act && !act[b]) && i < N) {
        float w = warped[(size_t)b * N + i];
        if (valid_vox(ref, mask, w, i)) {
            Stats s = get_stats(Bf + (size_t)b * 6 * N, N, i);
            float lv = (s.corr - (s.mr * s.mw)) / (s.sr * s.sw);
            if (lv == lv && !isinf(lv)) { sum = fabs((double)lv); cnt = 1.0; }
        }
    }
    block_sum2(sum, cnt);
    if (threadIdx.x == 0) {
        partial[((size_t)b * nblk + blockIdx.x) * 2] = sum;
        partial[((size_t)b * nblk + blockIdx.x) * 2 + 1] = cnt;
    }
}
__global__ void reduce_partials_k(const double *partial, int nblk, double *acc, int accOff, const int *act)
{
    int b = blockIdx.x;
    if (act && !act[b]) return;
    double s = 0.0, c = 0.0;
    for (int i = threadIdx.x; i < nblk; i += blockDim.x) { s += partial[((size_t)b * nblk + i) * 2]; c += partial[((size_t)b * nblk + i) * 2 + 1]; }
    block_sum2(s, c);
    if (threadIdx.x == 0) { acc[b * 4 + accOff] = s; acc[b * 4 + accOff + 1] = c; }
}
__global__ void value_final_k(const double *acc, double *out, int B, const int *act)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    if (act && !act[b]) return;
    out[b] = acc[b * 4] / acc[b * 4 + 1];
}

static void lncc_forward_stats(LnccCtx &c, const float *warped, const int *act, cudaStream_t st)
{
    dim3 g(cdiv(c.N, 256), c.B);
    lncc_prep_k<<<g, 256, 0, st>>>(c.ref, c.mask, warped, c.bufA, c.N, act);
    size_t bs = 6 * c.N;
    conv_axis(c.bufA, c.bufB, 6, c.B, bs, c.dims, 0, c.gt[0], st, c.exact, nullptr, act);
    conv_axis(c.bufB, c.bufA, 6, c.B, bs, c.dims, 1, c.gt[1], st, c.exact, nullptr, act);
    conv_axis(c.bufA, c.bufB, 6, c.B, bs, c.dims, 2, c.gt[2], st, c.exact, nullptr, act);
}

void reduce_partials(LnccCtx &c, int nblk, int accOff, const int *act, cudaStream_t st)
{
    reduce_partials_k<<<c.B, 256, 0, st>>>(c.partial, nblk, c.acc, accOff, act);
}
void finalize_value(LnccCtx &c, double *out, const int *act, cudaStream_t st)
{
    value_final_k<<<cdiv(c.B, 64), 64, 0, st>>>(c.acc, out, c.B, act);
}
void lncc_value(LnccCtx &c, const float *warped, double *out, const int *act, cudaStream_t st)
{
    if (c.fused) { lncc_value_fused(c, warped, out, act, c.exact, st); return; }
    lncc_forward_stats(c, warped, act, st);
    dim3 g(cdiv(c.N, 256), c.B);
    lncc_value_k<<<g, 256, 0, st>>>(c.ref, c.mask, warped, c.bufB, c.N, c.partial, c.nblk, act);
    reduce_partials_k<<<c.B, 256, 0, st>>>(c.partial, c.nblk, c.acc, 0, act);
    value_final_k<<<cdiv(c.B, 64), 64, 0, st>>>(c.acc, out, c.B, act);
}

// gradient temporaries (double precision like NiftyReg): store temp1..3 in A channels 1..3 ; count in partial
__global__ void lncc_gprep_k(const float *ref, const int *mask, const float *warped, const float *Bf, float *A, size_t N,
                             double *partial, int nblk, const int *act)
{
    int b = blockIdx.y;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double cnt = 0.0, dummy = 0.0;
    bool skip = (act && !act[b]);
    if (!skip && i < N) {
        float w = warped[(size_t)b * N + i];
        float *a = A + (size_t)b * 6 * N;
        float t1 = 0.f, t2 = 0.f, t3 = 0.f;
        if (valid_vox(ref, mask, w, i)) {
            Stats s = get_stats(Bf + (size_t)b * 6 * N, N, i);
            double refMean = s.mr, warMean = s.mw, refSdev = s.sr, warSdev = s.sw;
            double correla = (double)s.corr - (refMean * warMean);
            double temp1 = 1.0 / (refSdev * warSdev);
            double temp2 = correla / (refSdev * warSdev * warSdev * warSdev);
            double temp3 = (correla * warMean) / (refSdev * warSdev * warSdev * warSdev) - refMean / (refSdev * warSdev);
            if (temp1 == temp1 && !isinf(temp1) && temp2 == temp2 && !isinf(temp2) && temp3 == temp3 && !isinf(temp3)) {
                if (correla < 0) { temp1 *= -1.; temp2 *= -1.; temp3 *= -1.; }
                t1 = (float)temp1; t2 = (float)temp2; t3 = (float)temp3;
                cnt = 1.0;
            }
        }
        a[N + i] = t1; a[2 * N + i] = t2; a[3 * N + i] = t3;
    }
    block_sum2(cnt, dummy);
    if (threadIdx.x == 0) {
        partial[((size_t)b * nblk + blockIdx.x) * 2] = cnt;
        partial[((size_t)b * nblk + blockIdx.x) * 2 + 1] = 0.0;
    }
}

__global__ void lncc_gfinal_k(const float *ref, const int *mask, const float *warped, const float *wgrad, const float *Bf,
                              const float *Cf, float *voxgrad, size_t N, const double *acc, const int *act)
{
    int b = blockIdx.y;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (act && !act[b]) return;
    float w = warped[(size_t)b * N + i];
    float g0 = 0.f, g1 = 0.f, g2 = 0.f;
    if (valid_vox(ref, mask, w, i)) {
        const float *bb = Bf + (size_t)b * 6 * N;
        const float *cc = Cf + (size_t)b * 6 * N;
        float d = bb[i];
        float t1 = cc[i] / d, t2 = cc[N + i] / d, t3 = cc[2 * N + i] / d;
        float cm = t1 * ref[i] - t2 * w + t3;
        double common = (double)cm;
        double adjusted = 1.0 / acc[b * 4 + 2];
        common *= adjusted;
        const float *wg = wgrad + (size_t)b * 3 * N;
        g0 = (float)(0.0 - (double)wg[i] * common);
        g1 = (float)(0.0 - (double)wg[N + i] * common);
        g2 = (float)(0.0 - (double)wg[2 * N + i] * common);
        if (g0 != g0 || isinf(g0)) g0 = 0.f;
        if (g1 != g1 || isinf(g1)) g1 = 0.f;
        if (g2 != g2 || isinf(g2)) g2 = 0.f;
    }
    float *o = voxgrad + (size_t)b * 3 * N;
    o[i] = g0; o[N + i] = g1; o[2 * N + i] = g2;
}

void lncc_gradient(LnccCtx &c, const float *warped, const float *wgrad, float *voxgrad, const int *act, cudaStream_t st)
{
    c.gp_done = false;
    if (c.fused) lncc_stats_fused(c, warped, c.bufB, act, c.exact, st);
    else lncc_forward_stats(c, warped, act, st);  // result in bufB
    dim3 g(cdiv(c.N, 256), c.B);
    if (c.gp_done) {   // the fused kernel already produced the temporaries (bufA channels 1..3) and the counts (partial sums)
        reduce_partials_k<<<c.B, 256, 0, st>>>(c.partial, c.nblk_used, c.acc, 2, act);
    } else {
        lncc_gprep_k<<<g, 256, 0, st>>>(c.ref, c.mask, warped, c.bufB, c.bufA, c.N, c.partial, c.nblk, act);
        reduce_partials_k<<<c.B, 256, 0, st>>>(c.partial, c.nblk, c.acc, 2, act);
    }
    if (c.fused && lncc_gradient_smooth_fused(c, warped, wgrad, voxgrad, act, st)) return;
    size_t bs = 6 * c.N;
    // smooth temp1..3 (A channels 1..3) -> result in C channels 0..2
    conv_axis(c.bufA + c.N, c.bufC, 3, c.B, bs, c.dims, 0, c.gt[0], st, c.exact, nullptr, act);
    conv_axis(c.bufC, c.bufA + c.N, 3, c.B, bs, c.dims, 1, c.gt[1], st, c.exact, nullptr, act);
    conv_axis(c.bufA + c.N, c.bufC, 3, c.B, bs, c.dims, 2, c.gt[2], st, c.exact, nullptr, act);
    lncc_gfinal_k<<<g, 256, 0, st>>>(c.ref, c.mask, warped, wgrad, c.bufB, c.bufC, voxgrad, c.N, c.acc, act);
}

}  // namespace f3d
