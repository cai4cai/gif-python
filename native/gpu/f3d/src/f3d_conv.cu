// SPDX-License-Identifier: BSD-3-Clause
// Separable normalised convolution, image pyramid, intensity rescale.
#include "f3d_internal.h"
#include <map>
#include <mutex>
#include <cfloat>
#include <cmath>

namespace f3d {

// ---------------------------------------------------------------- device memory pool
static std::mutex g_pool_mu;
static std::map<void *, size_t> g_live;                       // allocated blocks
static std::multimap<size_t, void *> g_free;                   // cached free blocks by size
void *pool_malloc(size_t bytes)
{
    if (bytes == 0) bytes = 1;
    static const bool off = getenv("F3D_NOPOOL") != nullptr;
    void *p = nullptr;
    if (!off) {
        std::lock_guard<std::mutex> lk(g_pool_mu);
        auto it = g_free.lower_bound(bytes);
        if (it != g_free.end() && it->first <= bytes + bytes / 4 + (1 << 20)) {   // best fit within 25 %
            p = it->second;
            g_live[p] = it->first;
            g_free.erase(it);
            return p;
        }
    }
    CK(cudaMalloc(&p, bytes));
    if (!off) { std::lock_guard<std::mutex> lk(g_pool_mu); g_live[p] = bytes; }
    return p;
}
void pool_free(void *p)
{
    if (!p) return;
    static const bool off = getenv("F3D_NOPOOL") != nullptr;
    if (!off) {
        std::lock_guard<std::mutex> lk(g_pool_mu);
        auto it = g_live.find(p);
        if (it != g_live.end()) { g_free.insert({it->second, p}); g_live.erase(it); return; }
    }
    cudaFree(p);
}
void pool_trim()
{
    std::lock_guard<std::mutex> lk(g_pool_mu);
    for (auto &kv : g_free) cudaFree(kv.second);
    g_free.clear();
}

Taps make_taps(float sigma, float pixdim, int kind)
{
    Taps t;
    double temp;
    if (sigma > 0) { float q = sigma / pixdim; temp = q; }
    else temp = fabsf(sigma);
    int radius = (kind == 0) ? (int)(temp * 3.0f) : (int)(temp * 2.0f);
    t.radius = radius;
    if (radius > 0) {
        t.h.resize(2 * radius + 1);
        for (int i = -radius; i <= radius; i++) {
            if (kind == 1) {
                double relative = fabs((double)i / (double)temp);
                float v;
                if (relative < 1.0) v = (float)(2.0 / 3.0 - relative * relative + 0.5 * relative * relative * relative);
                else if (relative < 2.0) v = (float)(-(relative - 2.0) * (relative - 2.0) * (relative - 2.0) / 6.0);
                else v = 0;
                t.h[i + radius] = v;
            } else {
                t.h[radius + i] = (float)(exp(-(double)(i * i) / (2.0 * temp * temp)) / (temp * 2.506628274631));
            }
        }
        t.d = h2d(t.h.data(), t.h.size());
    }
    return t;
}
void free_taps(Taps &t)
{
    if (t.d) pool_free(t.d);
    t.d = nullptr;
}

std::vector<float> density_table(const Taps &t, int n)
{
    std::vector<float> d(n);
    for (int p = 0; p < n; p++) {
        double s = 0;
        int lo = p - t.radius < 0 ? 0 : p - t.radius, hi = p + t.radius > n - 1 ? n - 1 : p + t.radius;
        for (int k = lo; k <= hi; k++) s += (double)(t.h[k - p + t.radius] * 1.0f);
        d[p] = (float)s;
    }
    return d;
}

template <int C, bool EXACT>
__global__ void conv_axis_k(const float *__restrict__ in, float *__restrict__ out, size_t chStride, size_t bStride,
                            int nx, int ny, int nz, int axis, int radius, const float *__restrict__ ktab, const float *__restrict__ dnorm, const int *__restrict__ act,
                            const unsigned *__restrict__ list, int count)
{
    if (act && !act[blockIdx.y]) return;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t N = (size_t)nx * ny * nz;
    if (list) { if (idx >= (size_t)count) return; idx = list[idx]; }
    else if (idx >= N) return;
    in += (size_t)blockIdx.y * bStride;
    out += (size_t)blockIdx.y * bStride;
    int x = idx % nx, y = (idx / nx) % ny, z = idx / ((size_t)nx * ny);
    int p, n, stride;
    if (axis == 0) { p = x; n = nx; stride = 1; }
    else if (axis == 1) { p = y; n = ny; stride = nx; }
    else { p = z; n = nz; stride = nx * ny; }
    int lo = max(p - radius, 0), hi = min(p + radius, n - 1);
    float acc[C], err[C];
#pragma unroll
    for (int c = 0; c < C; c++) { acc[c] = 0.f; err[c] = 0.f; }
    for (int k = lo; k <= hi; k++) {
        float w = ktab[k - p + radius];
        const float *src = in + idx + (ptrdiff_t)(k - p) * stride;
#pragma unroll
        for (int c = 0; c < C; c++) {
            if (EXACT) {
                // NiftyReg: float product accumulated in double -> emulate with an error-free float two-sum
                float pr = __fmul_rn(w, src[c * chStride]);
                float s = acc[c] + pr;
                float bp = s - acc[c];
                err[c] += (acc[c] - (s - bp)) + (pr - bp);
                acc[c] = s;
            } else {
                acc[c] += w * src[c * chStride];
            }
        }
    }
#pragma unroll
    for (int c = 0; c < C; c++) {
        float r = EXACT ? (acc[c] + err[c]) : acc[c];
        if (dnorm) r = r / dnorm[p];
        out[c * chStride + idx] = r;
    }
}

void conv_axis(const float *in, float *out, int C, int B, size_t bStride, const int dims[3], int axis,
               const Taps &t, cudaStream_t st, bool exact, const float *dnorm, const int *act, const unsigned *list, int count)
{
    size_t N = (size_t)dims[0] * dims[1] * dims[2];
    dim3 grid(cdiv(list ? (size_t)count : N, 128), B);
    if (dims[axis] <= 1 || t.radius <= 0) {
        // NiftyReg skips the axis: plain copy
        if (in != out) {
            for (int b = 0; b < B; b++) CK(cudaMemcpyAsync(out + b * bStride, in + b * bStride, (size_t)C * N * sizeof(float), cudaMemcpyDeviceToDevice, st));
        }
        return;
    }
#define CASE(CC) case CC: if (exact) conv_axis_k<CC, true><<<grid, 128, 0, st>>>(in, out, N, bStride, dims[0], dims[1], dims[2], axis, t.radius, t.d, dnorm, act, list, count); else conv_axis_k<CC, false><<<grid, 128, 0, st>>>(in, out, N, bStride, dims[0], dims[1], dims[2], axis, t.radius, t.d, dnorm, act, list, count); break;
    switch (C) {
        CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
    default: fprintf(stderr, "conv_axis: unsupported channel count %d\n", C); exit(1);
    }
#undef CASE
}

// ---------------------------------------------------------------- pyramid
__global__ void blur_prep_k(const float *in, float *buf, size_t N)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = in[i];
    bool ok = (v == v);
    buf[i] = ok ? v : 0.f;
    buf[N + i] = ok ? 1.f : 0.f;
}
__global__ void downsample_k(const float *buf, float *out, size_t N, int nx, int ny, int nz, int ox, int oy, int oz,
                             int sx, int sy, int sz)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t No = (size_t)ox * oy * oz;
    if (i >= No) return;
    int x = i % ox, y = (i / ox) % oy, z = i / ((size_t)ox * oy);
    int X = x * sx, Y = y * sy, Z = z * sz;
    float v = nanf("");
    if (X < nx && Y < ny && Z < nz) {
        size_t j = ((size_t)Z * ny + Y) * nx + X;
        float d = buf[N + j];
        v = (d > 0.f) ? buf[j] / d : nanf("");
    }
    out[i] = v;
}

float *pyramid_step(const float *d_in, const ImgGeom &gin, ImgGeom &gout, cudaStream_t st)
{
    bool ax[3];
    downsample_axes(gin, ax);
    gout = geom_downsample(gin, ax);
    size_t N = gin.nvox();
    float *A = dalloc<float>(2 * N), *Bf = dalloc<float>(2 * N);
    blur_prep_k<<<cdiv(N, 256), 256, 0, st>>>(d_in, A, N);
    Taps t = make_taps(-0.7355f, 1.f, 0);
    // x,y,z passes; (A->B->A->B) result in Bf
    conv_axis(A, Bf, 2, 1, 2 * N, gin.n, 0, t, st);
    conv_axis(Bf, A, 2, 1, 2 * N, gin.n, 1, t, st);
    conv_axis(A, Bf, 2, 1, 2 * N, gin.n, 2, t, st);
    free_taps(t);
    float *out = dalloc<float>(gout.nvox());
    downsample_k<<<cdiv(gout.nvox(), 256), 256, 0, st>>>(Bf, out, N, gin.n[0], gin.n[1], gin.n[2], gout.n[0], gout.n[1],
                                                         gout.n[2], ax[0] ? 2 : 1, ax[1] ? 2 : 1, ax[2] ? 2 : 1);
    CK(cudaStreamSynchronize(st));
    pool_free(A); pool_free(Bf);
    return out;
}

void build_pyramid(const float *d_img, const ImgGeom &g0, int levelNumber, int levelToPerform,
                   std::vector<float *> &lv, std::vector<ImgGeom> &gl, cudaStream_t st)
{
    lv.assign(levelToPerform, nullptr);
    gl.assign(levelToPerform, g0);
    // finest performed level: input copy downsampled (levelNumber-levelToPerform) times
    const float *cur = d_img;
    ImgGeom cg = g0;
    float *owned = nullptr;
    for (int l = levelToPerform; l < levelNumber; l++) {
        ImgGeom ng;
        float *nxt = pyramid_step(cur, cg, ng, st);
        if (owned) pool_free(owned);
        owned = nxt; cur = nxt; cg = ng;
    }
    size_t n = cg.nvox();
    lv[levelToPerform - 1] = dalloc<float>(n);
    CK(cudaMemcpyAsync(lv[levelToPerform - 1], cur, n * sizeof(float), cudaMemcpyDeviceToDevice, st));
    gl[levelToPerform - 1] = cg;
    if (owned) pool_free(owned);
    for (int l = levelToPerform - 2; l >= 0; l--) {
        ImgGeom ng;
        lv[l] = pyramid_step(lv[l + 1], gl[l + 1], ng, st);
        gl[l] = ng;
    }
    CK(cudaStreamSynchronize(st));
}

// ---------------------------------------------------------------- rescale
__global__ void minmax_k(const float *d, size_t n, float *pmin, float *pmax)
{
    float mn = FLT_MAX, mx = -FLT_MAX;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float v = d[i];
        if (v == v) { mn = fminf(mn, v); mx = fmaxf(mx, v); }
    }
    __shared__ float smn[256], smx[256];
    smn[threadIdx.x] = mn; smx[threadIdx.x] = mx;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smn[threadIdx.x] = fminf(smn[threadIdx.x], smn[threadIdx.x + s]);
            smx[threadIdx.x] = fmaxf(smx[threadIdx.x], smx[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { pmin[blockIdx.x] = smn[0]; pmax[blockIdx.x] = smx[0]; }
}
__global__ void minmax_final_k(const float *pmin, const float *pmax, int nb, float *res)
{
    float mn = FLT_MAX, mx = -FLT_MAX;
    for (int i = threadIdx.x; i < nb; i += blockDim.x) { mn = fminf(mn, pmin[i]); mx = fmaxf(mx, pmax[i]); }
    __shared__ float smn[256], smx[256];
    smn[threadIdx.x] = mn; smx[threadIdx.x] = mx;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smn[threadIdx.x] = fminf(smn[threadIdx.x], smn[threadIdx.x + s]);
            smx[threadIdx.x] = fmaxf(smx[threadIdx.x], smx[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { res[0] = smn[0]; res[1] = smx[0]; }
}
__global__ void rescale_k(float *d, size_t n, const float *mm)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float mn = mm[0], mx = mm[1];
    double diff = (double)(mx - mn);
    double v = (double)d[i];
    if (v == v) { v = (v - (double)mn) / diff; v = v * 1.0 + 0.0; }
    d[i] = (float)v;
}
void rescale01(float *d, size_t n, cudaStream_t st)
{
    int nb = 64;
    float *tmp = dalloc<float>(2 * nb + 2);
    minmax_k<<<nb, 256, 0, st>>>(d, n, tmp, tmp + nb);
    minmax_final_k<<<1, 256, 0, st>>>(tmp, tmp + nb, nb, tmp + 2 * nb);
    rescale_k<<<cdiv(n, 256), 256, 0, st>>>(d, n, tmp + 2 * nb);
    CK(cudaStreamSynchronize(st));
    pool_free(tmp);
}

}  // namespace f3d
