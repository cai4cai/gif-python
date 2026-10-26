// SPDX-License-Identifier: BSD-3-Clause
// gif_stages.cu - see gif_stages.h.  Written from gif-python's Python code (src/multi_atlas/utils.py, multi_atlas_fusion.py,
// atlas_registration_and_resampling.py) and from the documented behaviour of NiftyReg's reg_resample and scipy.ndimage.gaussian_filter.
#include "gif_stages.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gifgpu {

using namespace f3d;

// ======================================================================================================== labels
__global__ void labels_from_float_k(const float *in, uint16_t *out, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(__float2half_rn(in[i]));   // float16 round trip: what get_fdata(dtype=np.float16) does
    out[i] = (v != v || v < 0.f) ? 0 : (v > 65535.f ? 65535 : (uint16_t)v);
}
void labels_from_float(const float *d_in, uint16_t *d_out, size_t n, cudaStream_t st)
{
    labels_from_float_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(d_in, d_out, n);
}
__global__ void label_presence_k(const uint16_t *lab, size_t n, unsigned char *present)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) present[lab[i]] = 1;
}
void label_presence(const uint16_t *d_lab, size_t n, unsigned char *d_present, cudaStream_t st)
{
    label_presence_k<<<1024, 256, 0, st>>>(d_lab, n, d_present);
}

// ======================================================================================================== Resampler
struct Resampler::Impl {
    cudaStream_t st;
    WarpArgs wa;
    AxisTab *tab[3] = {nullptr, nullptr, nullptr};
    size_t N = 0, Nf = 0, ndof = 0;
    int fn[3];
    float fijk[12];
    float *warped = nullptr, *def = nullptr;
};

// reg_round of NiftyReg (_reg_maths.h)
__device__ __forceinline__ int reg_round_d(double a) { return a > 0.0 ? (int)(a + 0.5) : (int)(a - 0.5); }

// result of the linear / nearest neighbour interpolation as stored by reg_resampleImage for the floating image data type
__device__ __forceinline__ float store_typed(double intensity, int dt)
{
    switch (dt) {
        case 16: case 64: return (float)intensity;
        case 2:   // uint8
            if (intensity != intensity) intensity = 0;
            intensity = (intensity <= 255 ? reg_round_d(intensity) : 255);
            return (float)(intensity > 0 ? reg_round_d(intensity) : 0);
        case 512:   // uint16
            if (intensity != intensity) intensity = 0;
            intensity = (intensity <= 65535 ? reg_round_d(intensity) : 65535);
            return (float)(intensity > 0 ? reg_round_d(intensity) : 0);
        case 768:   // uint32
            if (intensity != intensity) intensity = 0;
            intensity = (intensity <= 4294967295.0 ? (double)reg_round_d(intensity) : 4294967295.0);
            return (float)(intensity > 0 ? (double)reg_round_d(intensity) : 0.0);
        case 4:  { if (intensity != intensity) intensity = 0; return (float)(int16_t)reg_round_d(intensity); }
        case 256: { if (intensity != intensity) intensity = 0; return (float)(int8_t)reg_round_d(intensity); }
        default: { if (intensity != intensity) intensity = 0; return (float)reg_round_d(intensity); }
    }
}

// one voxel: position in the floating image from the deformation (reg_mat44_mul: double arithmetic, float result), then the
// separable kernel of reg_resampleImage (ResampleImage3D) with padding value 0
__global__ void interp_k(const float *__restrict__ def, size_t N, const float *__restrict__ flo_img, const uint16_t *__restrict__ flo_lab,
                         int nx, int ny, int nz, const float *fijk, int dt, float *out_img, uint16_t *out_lab)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float w0 = def[i], w1 = def[N + i], w2 = def[2 * N + i];
    float pos[3];
    for (int r = 0; r < 3; r++) {
        double s = __dmul_rn((double)fijk[r * 4 + 0], (double)w0);
        s = __dadd_rn(s, __dmul_rn((double)fijk[r * 4 + 1], (double)w1));
        s = __dadd_rn(s, __dmul_rn((double)fijk[r * 4 + 2], (double)w2));
        s = __dadd_rn(s, (double)fijk[r * 4 + 3]);
        pos[r] = (float)s;
    }
    int prev[3];
    double rel[3];
    for (int r = 0; r < 3; r++) {
        prev[r] = (int)floorf(pos[r]);
        rel[r] = (double)pos[r] - (double)prev[r];
        if (rel[r] < 0.0) rel[r] = 0.0;
    }
    if (out_img) {
        double xb[2] = {1.0 - rel[0], rel[0]}, yb[2] = {1.0 - rel[1], rel[1]}, zb[2] = {1.0 - rel[2], rel[2]};
        double inten = 0.0;
        for (int c = 0; c < 2; c++) {
            const int Z = prev[2] + c;
            double yt = 0.0;
            for (int b = 0; b < 2; b++) {
                const int Y = prev[1] + b;
                double xt = 0.0;
                for (int a = 0; a < 2; a++) {
                    const int X = prev[0] + a;
                    const bool in = (X > -1 && X < nx && Y > -1 && Y < ny && Z > -1 && Z < nz);
                    const double v = in ? (double)flo_img[((size_t)Z * ny + Y) * nx + X] : 0.0;
                    xt = __dadd_rn(xt, __dmul_rn(v, xb[a]));
                }
                yt = __dadd_rn(yt, __dmul_rn(xt, yb[b]));
            }
            inten = __dadd_rn(inten, __dmul_rn(yt, zb[c]));
        }
        out_img[i] = store_typed(inten, dt);
    }
    if (out_lab) {
        const int X = prev[0] + (rel[0] >= 0.5 ? 1 : 0), Y = prev[1] + (rel[1] >= 0.5 ? 1 : 0), Z = prev[2] + (rel[2] >= 0.5 ? 1 : 0);
        out_lab[i] = (X > -1 && X < nx && Y > -1 && Y < ny && Z > -1 && Z < nz) ? flo_lab[((size_t)Z * ny + Y) * nx + X] : 0;
    }
}

Resampler::Resampler(const ImgGeom &ref, const GridGeom &grid, const ImgGeom &flo, cudaStream_t st)
{
    p_ = new Impl;
    Impl &I = *p_;
    I.st = st;
    I.N = ref.nvox(); I.Nf = flo.nvox(); I.ndof = 3 * grid.nnodes();
    memset(&I.wa, 0, sizeof I.wa);
    for (int a = 0; a < 3; a++) { I.wa.rn[a] = ref.n[a]; I.wa.gn[a] = grid.n[a]; I.wa.fn[a] = flo.n[a]; I.fn[a] = flo.n[a]; }
    for (int r = 0; r < 3; r++) for (int c = 0; c < 4; c++) { I.wa.fijk[r * 4 + c] = flo.ijk.m[r][c]; I.fijk[r * 4 + c] = flo.ijk.m[r][c]; }
    float gvs[3];
    for (int a = 0; a < 3; a++) gvs[a] = grid.d[a] / ref.d[a];
    const bool lut = (gvs[0] == 5.f);
    std::vector<AxisTab> htab[3];
    for (int a = 0; a < 3; a++) {
        htab[a] = lut ? build_tab_lut(ref.n[a], grid.n[a]) : build_tab_generic(ref.n[a], gvs[a]);
        I.tab[a] = h2d(htab[a].data(), htab[a].size());
        I.wa.tab[a] = I.tab[a];
    }
    warp_prepare(I.wa, htab);
    I.wa.cflag = nullptr;
    I.wa.fast = false;
    I.wa.mask = nullptr;
    I.warped = dalloc<float>(I.N);
    I.def = dalloc<float>(3 * I.N);
}
Resampler::~Resampler()
{
    for (int a = 0; a < 3; a++) pool_free(p_->tab[a]);
    pool_free(p_->warped); pool_free(p_->def);
    delete p_;
}
void Resampler::run(const float *d_cpp, const float *d_flo_img, int dt, float *d_img_out, const uint16_t *d_flo_lab, uint16_t *d_lab_out)
{
    Impl &I = *p_;
    warp(I.wa, 1, d_cpp, I.ndof, d_flo_img, I.Nf, I.warped, nullptr, I.def, nullptr, I.st);
    float *dfijk = dalloc<float>(12);
    CK(cudaMemcpyAsync(dfijk, I.fijk, 12 * sizeof(float), cudaMemcpyHostToDevice, I.st));
    interp_k<<<(unsigned)((I.N + 127) / 128), 128, 0, I.st>>>(I.def, I.N, d_flo_img, d_flo_lab, I.fn[0], I.fn[1], I.fn[2], dfijk, dt,
                                                              d_img_out, d_lab_out);
    CK(cudaStreamSynchronize(I.st));
    pool_free(dfijk);
}

// ======================================================================================================== LNCC weights
namespace {
constexpr int kMaxRadius = 40;

// numpy's float64 pairwise sum (used by scipy's _gaussian_kernel1d for phi_x.sum())
double np_sum_f64(const double *a, int n)
{
    if (n < 8) {
        double res = 0.;
        for (int i = 0; i < n; i++) res += a[i];
        return res;
    }
    double r[8];
    for (int j = 0; j < 8; j++) r[j] = a[j];
    int i;
    for (i = 8; i < n - (n % 8); i += 8)
        for (int j = 0; j < 8; j++) r[j] += a[i + j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[i];
    return res;
}

struct SmoothArgs {
    int n[3];
    int radius[3];
    const double *taps[3];   // [radius+1] per axis, taps[0] = centre
};

// one separable pass along AXIS with scipy's symmetric correlate1d accumulation order (centre first, then the pairs from the outside
// in), double accumulation, reflect ('d c b a | a b c d | d c b a') boundary, float output.
// SRC: 0 plain, 1 = atlas*atlas, 2 = target*atlas, 3 = target*target (the quantities are formed in float like numpy does)
template <int AXIS, int SRC>
__global__ void smooth_pass_k(SmoothArgs A, const float *__restrict__ in, const float *__restrict__ img, float *__restrict__ out)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t N = (size_t)A.n[0] * A.n[1] * A.n[2];
    if (i >= N) return;
    const int x = i % A.n[0], y = (i / A.n[0]) % A.n[1], z = i / ((size_t)A.n[0] * A.n[1]);
    const int pos = AXIS == 0 ? x : (AXIS == 1 ? y : z);
    const int len = A.n[AXIS];
    const size_t stride = AXIS == 0 ? 1 : (AXIS == 1 ? (size_t)A.n[0] : (size_t)A.n[0] * A.n[1]);
    const size_t base = i - (size_t)pos * stride;
    const int r = A.radius[AXIS];
    const double *w = A.taps[AXIS];
    auto val = [&](int p) -> double {
        if (p < 0) p = -p - 1;
        if (p >= len) p = 2 * len - p - 1;
        const size_t j = base + (size_t)p * stride;
        float v = in[j];
        if (SRC == 1) v = __fmul_rn(v, v);
        else if (SRC == 2) v = __fmul_rn(img[j], v);
        else if (SRC == 3) v = __fmul_rn(v, v);
        return (double)v;
    };
    double acc = __dmul_rn(val(pos), w[0]);
    for (int k = r; k >= 1; k--) acc = __dadd_rn(acc, __dmul_rn(__dadd_rn(val(pos - k), val(pos + k)), w[k]));
    out[i] = (float)acc;
}

__global__ void weights_k(size_t N, const float *mi, const float *si, const float *ma, const float *sa, const float *pr, float temp,
                          float *out)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float imean = mi[i], amean = ma[i];
    const float istd = __fsub_rn(si[i], __fmul_rn(imean, imean));
    const float astd = __fsub_rn(sa[i], __fmul_rn(amean, amean));
    const float covar = __fsub_rn(pr[i], __fmul_rn(imean, amean));
    float vp = __fsqrt_rn(__fmul_rn(istd, astd));
    if (vp == 0.f) vp = nanf("");
    const float lncc = __fdiv_rn(covar, vp);
    const float dist = __fsub_rn(1.0f, lncc);
    const float wv = expf(__fdiv_rn(-dist, temp));
    out[i] = (wv != wv) ? 0.f : wv;
}
}  // namespace

struct LnccWeights::Impl {
    cudaStream_t st;
    SmoothArgs A;
    size_t N;
    float temp;
    double sigma[3];
    bool ok = true;
    double *dtaps[3] = {nullptr, nullptr, nullptr};
    const float *img;
    float *mi, *si, *ma, *sa, *pr, *t1, *t2;
};

LnccWeights::LnccWeights(const float *d_target, const int dims[3], const float sigma_cfg[3], const float spacing[3], float temperature,
                         cudaStream_t st)
{
    p_ = new Impl;
    Impl &I = *p_;
    I.st = st; I.img = d_target; I.temp = temperature;
    I.N = (size_t)dims[0] * dims[1] * dims[2];
    for (int a = 0; a < 3; a++) {
        I.A.n[a] = dims[a];
        // kernel_std = abs(k / s) if k < 0 else k   (float32 arithmetic: python float with a float32 numpy scalar)
        float s = sigma_cfg[a] < 0.f ? fabsf(sigma_cfg[a] / spacing[a]) : sigma_cfg[a];
        I.sigma[a] = (double)s;
        I.A.radius[a] = (int)std::floor(3.0 * (double)s);
        if (I.A.radius[a] > kMaxRadius || I.A.radius[a] >= dims[a]) { I.ok = false; I.A.radius[a] = 0; }
        const int r = I.A.radius[a];
        std::vector<double> phi(2 * r + 1);
        const double sigma2 = (double)s * (double)s;
        for (int k = -r; k <= r; k++) phi[k + r] = std::exp(-0.5 / sigma2 * (double)(k * k));
        const double sum = np_sum_f64(phi.data(), 2 * r + 1);
        std::vector<double> half(r + 1);
        for (int k = 0; k <= r; k++) half[k] = phi[r + k] / sum;
        I.dtaps[a] = h2d(half.data(), half.size());
        I.A.taps[a] = I.dtaps[a];
    }
    I.mi = dalloc<float>(I.N); I.si = dalloc<float>(I.N); I.ma = dalloc<float>(I.N); I.sa = dalloc<float>(I.N); I.pr = dalloc<float>(I.N);
    I.t1 = dalloc<float>(I.N); I.t2 = dalloc<float>(I.N);
    // template independent quantities: smoothed target and smoothed squared target
    const unsigned g = (unsigned)((I.N + 255) / 256);
    smooth_pass_k<0, 0><<<g, 256, 0, st>>>(I.A, d_target, d_target, I.t1);
    smooth_pass_k<1, 0><<<g, 256, 0, st>>>(I.A, I.t1, d_target, I.t2);
    smooth_pass_k<2, 0><<<g, 256, 0, st>>>(I.A, I.t2, d_target, I.mi);
    smooth_pass_k<0, 3><<<g, 256, 0, st>>>(I.A, d_target, d_target, I.t1);
    smooth_pass_k<1, 0><<<g, 256, 0, st>>>(I.A, I.t1, d_target, I.t2);
    smooth_pass_k<2, 0><<<g, 256, 0, st>>>(I.A, I.t2, d_target, I.si);
    CK(cudaStreamSynchronize(st));
}
LnccWeights::~LnccWeights()
{
    Impl &I = *p_;
    for (int a = 0; a < 3; a++) pool_free(I.dtaps[a]);
    pool_free(I.mi); pool_free(I.si); pool_free(I.ma); pool_free(I.sa); pool_free(I.pr); pool_free(I.t1); pool_free(I.t2);
    delete p_;
}
bool LnccWeights::ok() const { return p_->ok; }
const double *LnccWeights::sigma() const { return p_->sigma; }
void LnccWeights::run(const float *d_war, float *d_weights)
{
    Impl &I = *p_;
    const unsigned g = (unsigned)((I.N + 255) / 256);
    cudaStream_t st = I.st;
    smooth_pass_k<0, 0><<<g, 256, 0, st>>>(I.A, d_war, I.img, I.t1);
    smooth_pass_k<1, 0><<<g, 256, 0, st>>>(I.A, I.t1, I.img, I.t2);
    smooth_pass_k<2, 0><<<g, 256, 0, st>>>(I.A, I.t2, I.img, I.ma);
    smooth_pass_k<0, 1><<<g, 256, 0, st>>>(I.A, d_war, I.img, I.t1);
    smooth_pass_k<1, 0><<<g, 256, 0, st>>>(I.A, I.t1, I.img, I.t2);
    smooth_pass_k<2, 0><<<g, 256, 0, st>>>(I.A, I.t2, I.img, I.sa);
    smooth_pass_k<0, 2><<<g, 256, 0, st>>>(I.A, d_war, I.img, I.t1);
    smooth_pass_k<1, 0><<<g, 256, 0, st>>>(I.A, I.t1, I.img, I.t2);
    smooth_pass_k<2, 0><<<g, 256, 0, st>>>(I.A, I.t2, I.img, I.pr);
    weights_k<<<g, 256, 0, st>>>(I.N, I.mi, I.si, I.ma, I.sa, I.pr, I.temp, d_weights);
}

// ======================================================================================================== Fusion
namespace {

struct FuseTab {
    size_t N;
    int A, nclass, ntissue;
    int nx, ny, nz;
    const uint16_t *const *lab;
    const float *const *w;
    const int16_t *lut;      // label -> class index
    const uint16_t *labels;  // class -> label
    const uint8_t *ntis;     // [nclass]
    const uint8_t *tis;      // [nclass][kMaxTissues]
    const uint16_t *tmask;   // [nclass] bit t set: the class may be part of tissue t
};

// numpy's float32 pairwise sum along a short contiguous axis
__device__ __forceinline__ float np_sum_f32(const float *a, int n)
{
    if (n < 8) {
        float res = 0.f;
        for (int i = 0; i < n; i++) res = __fadd_rn(res, a[i]);
        return res;
    }
    float r[8];
    for (int j = 0; j < 8; j++) r[j] = a[j];
    int i;
    for (i = 8; i < n - (n % 8); i += 8)
        for (int j = 0; j < 8; j++) r[j] = __fadd_rn(r[j], a[i + j]);
    float res = __fadd_rn(__fadd_rn(__fadd_rn(r[0], r[1]), __fadd_rn(r[2], r[3])), __fadd_rn(__fadd_rn(r[4], r[5]), __fadd_rn(r[6], r[7])));
    for (; i < n; i++) res = __fadd_rn(res, a[i]);
    return res;
}

// sparse label probabilities of voxel i: classes (ascending index) and sums of the weights of the atlases that carry the class.
// Reproduces get_multi_atlas_proba_seg: the sum over the atlases is accumulated in atlas order in float32; where no atlas has a non-zero
// label, or all sums are zero, class 0 gets probability 1.
__device__ int fuse_voxel(const FuseTab &F, size_t i, int *cls, float *sum)
{
    bool any = false;
    for (int a = 0; a < F.A; a++) if (F.lab[a][i] != 0) { any = true; break; }
    if (!any) { cls[0] = 0; sum[0] = 1.f; return 1; }
    int k = 0;
    for (int a = 0; a < F.A; a++) {
        const int c = F.lut[F.lab[a][i]];
        if (c < 0) continue;
        float wv = F.w[a][i];
        if (wv != wv) wv = 0.f;
        int j = 0;
        while (j < k && cls[j] != c) j++;
        if (j == k) { cls[k] = c; sum[k] = 0.f; k++; }
        sum[j] = __fadd_rn(sum[j], wv);
    }
    float total = 0.f;
    for (int j = 0; j < k; j++) total = __fadd_rn(total, sum[j]);
    if (k == 0 || total == 0.f) {
        int j = 0;
        while (j < k && cls[j] != 0) j++;
        if (j == k) { cls[k] = 0; k++; }
        sum[j] = 1.f;
    }
    for (int a = 1; a < k; a++) {   // insertion sort by class index
        int c = cls[a]; float s = sum[a]; int b = a - 1;
        while (b >= 0 && cls[b] > c) { cls[b + 1] = cls[b]; sum[b + 1] = sum[b]; b--; }
        cls[b + 1] = c; sum[b + 1] = s;
    }
    return k;
}

__global__ void argmax_k(FuseTab F, uint16_t *out)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= F.N) return;
    int cls[kMaxAtlases + 1]; float sum[kMaxAtlases + 1];
    const int k = fuse_voxel(F, i, cls, sum);
    int best = 0;
    for (int j = 1; j < k; j++) if (sum[j] > sum[best]) best = j;
    out[i] = F.labels[cls[best]];
}

__global__ void prior_k(FuseTab F, float *prior)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= F.N) return;
    int cls[kMaxAtlases + 1]; float sum[kMaxAtlases + 1];
    const int k = fuse_voxel(F, i, cls, sum);
    const int T = F.ntissue;
    float p[kMaxTissues];
    for (int t = 0; t < T; t++) p[t] = 0.f;
    for (int j = 0; j < k; j++) {
        const int c = cls[j], nt = F.ntis[c];
        if (nt == 0) continue;
        const float v = __fdiv_rn(sum[j], (float)nt);
        for (int q = 0; q < nt; q++) { const int t = F.tis[c * kMaxTissues + q]; p[t] = __fadd_rn(p[t], v); }
    }
    float s = np_sum_f32(p, T);
    if (s == 0.f) { p[0] = 1.f; s = np_sum_f32(p, T); }
    for (int t = 0; t < T; t++) p[t] = __fdiv_rn(p[t], s);
    for (int t = 0; t < T; t++) if (p[t] < 0.01f) p[t] = 0.f;
    s = np_sum_f32(p, T);
    if (s == 0.f) p[0] = 1.f;
    s = np_sum_f32(p, T);
    for (int t = 0; t < T; t++) prior[(size_t)t * F.N + i] = __fdiv_rn(p[t], s);
}

__global__ void tseg_k(FuseTab F, const float *post, const float *prior, uint8_t *tseg)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= F.N) return;
    const int x = i % F.nx, y = (i / F.nx) % F.ny, z = i / ((size_t)F.nx * F.ny);
    const bool border = (x == 0 || x == F.nx - 1 || y == 0 || y == F.ny - 1 || z == 0 || z == F.nz - 1);
    if (border || prior[i] == 1.f) { tseg[i] = 0; return; }   // every component is set to 1: argmax is the first class
    int best = 0;
    float bv = __half2float(__float2half_rn(post[i]));
    for (int t = 1; t < F.ntissue; t++) {
        const float v = __half2float(__float2half_rn(post[(size_t)t * F.N + i]));
        if (v > bv) { bv = v; best = t; }
    }
    tseg[i] = (uint8_t)best;
}

// get_structure_seg_from_tissue_seg: labels in order of decreasing probability, first one whose tissues contain the tissue label
__global__ void final_k(FuseTab F, const uint8_t *tseg, int32_t *out)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= F.N) return;
    int cls[kMaxAtlases + 1]; float sum[kMaxAtlases + 1];
    const int k = fuse_voxel(F, i, cls, sum);
    const unsigned tb = 1u << tseg[i];
    unsigned long long used = 0ull;
    int result = -1;
    for (int it = 0; it < k; it++) {
        int best = -1;
        for (int j = 0; j < k; j++) {
            if ((used >> j) & 1ull) continue;
            if (sum[j] > 0.f && (best < 0 || sum[j] > sum[best])) best = j;
        }
        if (best < 0) break;
        used |= 1ull << best;
        if (F.tmask[cls[best]] & tb) { result = F.labels[cls[best]]; break; }
    }
    if (result < 0) {
        // labels with probability zero follow in class order (the order numpy's argpartition returns for ties is not reproduced)
        for (int c = 0; c < F.nclass && result < 0; c++) {
            if (!(F.tmask[c] & tb)) continue;
            bool nonzero = false;
            for (int j = 0; j < k; j++) if (cls[j] == c && sum[j] > 0.f) nonzero = true;
            if (!nonzero) result = F.labels[c];
        }
    }
    out[i] = result < 0 ? 0 : result;
}
}  // namespace

struct Fusion::Impl {
    cudaStream_t st;
    FuseTab F;
    int16_t *lut;
    uint16_t *labels, *tmask;
    uint8_t *ntis, *tis;
    const uint16_t **lab = nullptr;
    const float **w = nullptr;
};

Fusion::Fusion(size_t N, int dims[3], const std::vector<uint16_t> &labels, const std::vector<std::vector<int>> &tissues, int ntissue,
               cudaStream_t st)
{
    p_ = new Impl;
    Impl &I = *p_;
    I.st = st;
    memset(&I.F, 0, sizeof I.F);
    I.F.N = N; I.F.nx = dims[0]; I.F.ny = dims[1]; I.F.nz = dims[2];
    I.F.nclass = (int)labels.size(); I.F.ntissue = ntissue;
    std::vector<int16_t> lut(65536, -1);
    for (size_t c = 0; c < labels.size(); c++) lut[labels[c]] = (int16_t)c;
    std::vector<uint8_t> ntis(labels.size(), 0), tis(labels.size() * kMaxTissues, 0);
    std::vector<uint16_t> tmask(labels.size(), 0);
    for (size_t c = 0; c < labels.size(); c++) {
        ntis[c] = (uint8_t)tissues[c].size();
        for (size_t q = 0; q < tissues[c].size(); q++) { tis[c * kMaxTissues + q] = (uint8_t)tissues[c][q]; tmask[c] |= (uint16_t)(1u << tissues[c][q]); }
    }
    I.lut = h2d(lut.data(), lut.size());
    I.labels = h2d(labels.data(), labels.size());
    I.ntis = h2d(ntis.data(), ntis.size());
    I.tis = h2d(tis.data(), tis.size());
    I.tmask = h2d(tmask.data(), tmask.size());
    I.F.lut = I.lut; I.F.labels = I.labels; I.F.ntis = I.ntis; I.F.tis = I.tis; I.F.tmask = I.tmask;
}
Fusion::~Fusion()
{
    Impl &I = *p_;
    pool_free(I.lut); pool_free(I.labels); pool_free(I.ntis); pool_free(I.tis); pool_free(I.tmask);
    if (I.lab) pool_free(I.lab);
    if (I.w) pool_free(I.w);
    delete p_;
}
void Fusion::set_inputs(const std::vector<const uint16_t *> &lab, const std::vector<const float *> &w)
{
    Impl &I = *p_;
    I.F.A = (int)lab.size();
    if (I.lab) pool_free(I.lab);
    if (I.w) pool_free(I.w);
    I.lab = (const uint16_t **)h2d(lab.data(), lab.size());
    I.w = (const float **)h2d(w.data(), w.size());
    I.F.lab = I.lab; I.F.w = I.w;
}
void Fusion::argmax_labels(uint16_t *d_out) { argmax_k<<<(unsigned)((p_->F.N + 127) / 128), 128, 0, p_->st>>>(p_->F, d_out); }
void Fusion::tissue_prior(float *d_prior) { prior_k<<<(unsigned)((p_->F.N + 127) / 128), 128, 0, p_->st>>>(p_->F, d_prior); }
void Fusion::tissue_seg(const float *d_post, const float *d_prior, uint8_t *d_tseg)
{
    tseg_k<<<(unsigned)((p_->F.N + 127) / 128), 128, 0, p_->st>>>(p_->F, d_post, d_prior, d_tseg);
}
void Fusion::final_parcellation(const uint8_t *d_tseg, int32_t *d_out)
{
    final_k<<<(unsigned)((p_->F.N + 127) / 128), 128, 0, p_->st>>>(p_->F, d_tseg, d_out);
}

}  // namespace gifgpu
