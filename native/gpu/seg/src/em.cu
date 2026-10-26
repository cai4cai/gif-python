// SPDX-License-Identifier: BSD-3-Clause
// em.cu - NiftySeg seg_EM (single modality, MRF, polynomial bias field) on the GPU.
//
// Iteration (identical order to seg_EM::Run_EM):  Maximization -> Expectation -> MRF -> BiasField -> (relaxation: off)
//   * Maximization: per class sums  S0=sum e, S1=sum e*y, S2=sum e*y^2  (y = T + bias) accumulated in DOUBLE with a
//     fixed (deterministic) reduction tree; M = S1/S0, V = sum e (y-M)^2 / S0 evaluated from the moments.
//   * Expectation: p_c = prior_c * N(y; M_c, V_c), normalised; loglik accumulated in double.
//     prior_c is the MRF field (initially 1, exactly like the CPU: the very first E-step ignores the atlas prior).
//   * MRF: 6-neighbourhood, exp(-beta * (sum over the other classes of neighbour posteriors)) * prior. Neighbours outside
//     the mask alias to the first voxel of the mask (a quirk of the CPU's short-index lookup), border voxels are not updated.
//   * Bias field: weighted least squares for a 3D polynomial of order 4 (35 coefficients) on every 2nd voxel per axis.
//     The normal equations A'WA (35x35) and A'Wr are built from separable power sums (line sums + 165 moments) in double,
//     and solved in double on the device.
// All state (M, V, loglik, iteration, coefficients) lives on the device; the host only syncs once per iteration to
// evaluate the stopping rule.
#include "internal.cuh"
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>

namespace segem {

// ------------------------------------------------------------------ helpers
__constant__ uint8_t c_mom[kMaxMoments][3];     // (p,q,r) of the AWA moments
__constant__ uint8_t c_bas[kMaxBasis][3];       // (a,b,c) of the basis functions (CPU ordering)
__constant__ uint8_t c_midx[9][9][9];           // (p,q,r) -> moment index

struct VP { int x, y, z; size_t v; };
__device__ __forceinline__ VP vp_decode(int v, int nx, int ny) {
    VP r;
    r.v = (size_t)v;
    r.x = v % nx;
    int q = v / nx;
    r.y = q % ny;
    r.z = q / ny;
    return r;
}

// polynomial bias field  b = - sum_k coef[k] * x^a y^b z^c   (float, same operation order as seg_EM)
__device__ __forceinline__ float bias_eval(const float *coef, int order, float xpos, float ypos, float zpos) {
    float xp[kMaxOrder + 1], yp[kMaxOrder + 1], zp[kMaxOrder + 1];
    xp[0] = yp[0] = zp[0] = 1.f;
#pragma unroll
    for (int o = 1; o <= kMaxOrder; o++) {
        xp[o] = __fmul_rn(xp[o - 1], xpos);
        yp[o] = __fmul_rn(yp[o - 1], ypos);
        zp[o] = __fmul_rn(zp[o - 1], zpos);
    }
    float b = 0.f;
    int ind = 0;
#pragma unroll
    for (int o = 0; o <= kMaxOrder; o++) {
#pragma unroll
        for (int xo = 0; xo <= o; xo++) {
#pragma unroll
            for (int yo = 0; yo <= o - xo; yo++) {
                int zo = o - yo - xo;
                if (o <= order) b = __fsub_rn(b, __fmul_rn(__fmul_rn(__fmul_rn(coef[ind], xp[xo]), yp[yo]), zp[zo]));
                ind++;
            }
        }
    }
    return b;
}

struct Geo {
    int nx, ny, nz, ntx, order;
    float hx, hy, hz, ihx, ihy, ihz;   // 0.5*n and 1/(0.5*n)
    size_t N;
};
static Geo make_geo(const EmContext::Impl &I) {
    Geo g;
    g.nx = I.d.nx; g.ny = I.d.ny; g.nz = I.d.nz; g.ntx = I.d.ntilesX; g.order = I.prm.bias_order; g.N = I.d.N;
    g.hx = 0.5f * (float)g.nx; g.hy = 0.5f * (float)g.ny; g.hz = 0.5f * (float)g.nz;
    g.ihx = 1.0f / g.hx; g.ihy = 1.0f / g.hy; g.ihz = 1.0f / g.hz;
    return g;
}
__device__ __forceinline__ float posf(int i, float h, float ih) { return ((float)i - h) * ih; }

// normalised prior of one voxel (seg_EM::InitializeAndNormalizeNaNPriors, applied on the fly)
template <int NC>
__device__ __forceinline__ void load_prior_norm(const float *prior, size_t N, size_t v, int C, float *p) {
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < NC; c++) {
        float x = 0.f;
        if (c < C) {
            x = prior[(size_t)c * N + v];
            if (x < 0.0f || x != x || x > 1000) x = 0.0f;
            s += x;
        }
        p[c] = x;
    }
    if (s > 0 && s < 1000) {
#pragma unroll
        for (int c = 0; c < NC; c++) p[c] = p[c] / s;
    } else {
#pragma unroll
        for (int c = 0; c < NC; c++) p[c] = 1.0f / (float)C;
    }
}

template <int R>
__device__ __forceinline__ void block_reduce_store(double (&val)[R], double *smem /*[4][R]*/, double *out) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
    for (int r = 0; r < R; r++) {
        double x = val[r];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
        if (lane == 0) smem[w * R + r] = x;
    }
    __syncthreads();
    if (threadIdx.x < R) {
        double s = smem[threadIdx.x];
        s += smem[R + threadIdx.x];
        s += smem[2 * R + threadIdx.x];
        s += smem[3 * R + threadIdx.x];
        out[(size_t)blockIdx.x * R + threadIdx.x] = s;
    }
}

// ------------------------------------------------------------------ voxel list, statistics, normalisation
struct IsMasked {
    const uint8_t *mask;
    __device__ bool operator()(const int &i) const { return mask[i] != 0; }
};
struct IsBorder {   // voxels of the list that are on the border of the volume (not updated by the MRF step)
    int nx, ny, nz;
    __device__ bool operator()(const int &i) const {
        int x = i % nx, y = (i / nx) % ny, z = i / (nx * ny);
        return !(x >= 1 && x <= nx - 2 && y >= 1 && y <= ny - 2 && z >= 1 && z <= nz - 2);
    }
};

// min / max of the target inside the mask (grid-stride over the list), per-block partials
__global__ void k_list_stats(const int *vox, const int *nvox, const float *target, float *pmin, float *pmax) {
    int n = *nvox;
    float mn = 1e32f, mx = -1e32f;   // same compare pattern as seg_EM (NaN never updates)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float x = target[vox[i]];
        mn = x < mn ? x : mn;
        mx = x > mx ? x : mx;
    }
    __shared__ float smn[kTileX], smx[kTileX];
    smn[threadIdx.x] = mn; smx[threadIdx.x] = mx;
    __syncthreads();
    for (int s = kTileX / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { smn[threadIdx.x] = fminf(smn[threadIdx.x], smn[threadIdx.x + s]); smx[threadIdx.x] = fmaxf(smx[threadIdx.x], smx[threadIdx.x + s]); }
        __syncthreads();
    }
    if (threadIdx.x == 0) { pmin[blockIdx.x] = smn[0]; pmax[blockIdx.x] = smx[0]; }
}

// one block: totals; resets the EM state
__global__ void k_list_final(const int *vox, const int *nvox, const float *pmin, const float *pmax, int nb, EMState *st) {
    float mn = 1e32f, mx = -1e32f;
    for (int i = threadIdx.x; i < nb; i += blockDim.x) { mn = fminf(mn, pmin[i]); mx = fmaxf(mx, pmax[i]); }
    __shared__ float smn[256], smx[256];
    smn[threadIdx.x] = mn; smx[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { smn[threadIdx.x] = fminf(smn[threadIdx.x], smn[threadIdx.x + s]); smx[threadIdx.x] = fmaxf(smx[threadIdx.x], smx[threadIdx.x + s]); }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        int n = *nvox;
        st->numMasked = n; st->rmin = smn[0]; st->rmax = smx[0]; st->firstMasked = n > 0 ? vox[0] : 0;
        for (int c = 0; c < kMaxClasses; c++) { st->M[c] = 0.f; st->V[c] = 0.f; st->inv_v[c] = 0.f; st->inv_sqrt[c] = 0.f; }
        for (int k = 0; k < kMaxBasis; k++) st->coef[k] = 0.f;
        st->loglik = 2.0; st->oldloglik = 1.0; st->iter = 0; st->biasActive = 0; st->biasRan = 0;
    }
}

// seg_EM::InitializeAndNormalizeImage: T = log2(1 + (x-min)/(max-min))  (logf in float, divide by the double constant)
__global__ void k_normalize(const float *target, float *T, const int *vox, const int *nvox, const EMState *st, double *pT) {
    int n = *nvox;
    float tmin = st->rmin, tmax = st->rmax;
    double s1 = 0, s2 = 0, cnt = 0;       // sum T, T^2, #valid (used by the closed-form first M-step, see k_m_uniform)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        int v = vox[i];
        float x = target[v];
        float t = (float)((double)logf((((x) - tmin) / (tmax - tmin)) + 1) / 0.693147181);
        T[v] = t;
        if (t == t) { s1 += (double)t; s2 += (double)t * (double)t; cnt += 1; }
    }
    __shared__ double sm[3 * 8];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    double val[3] = {cnt, s1, s2};
#pragma unroll
    for (int r = 0; r < 3; r++) {
        double xv = val[r];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) xv += __shfl_down_sync(0xffffffffu, xv, o);
        if (lane == 0) sm[w * 3 + r] = xv;
    }
    __syncthreads();
    if (threadIdx.x < 3) {
        double sres = 0;
        for (int k = 0; k < (int)(blockDim.x / 32); k++) sres += sm[k * 3 + threadIdx.x];
        pT[(size_t)blockIdx.x * 3 + threadIdx.x] = sres;
    }
}

// Closed-form M-step of iteration 1 (dormant classes exist): the iteration-0 posterior is the uniform 1/C everywhere, so every
// class has the same sums  S0 = n u, S1 = u sum(T), S2 = u sum(T^2).  Writes them as a single partial block for k_mfinal.
template <int NC>
__global__ void k_m_uniform(const double *pT, int nb, int C, double *part) {
    __shared__ double sm[256 * 3];
    double c = 0, a = 0, b = 0;
    for (int i = threadIdx.x; i < nb; i += blockDim.x) { c += pT[(size_t)i * 3]; a += pT[(size_t)i * 3 + 1]; b += pT[(size_t)i * 3 + 2]; }
    sm[threadIdx.x * 3] = c; sm[threadIdx.x * 3 + 1] = a; sm[threadIdx.x * 3 + 2] = b;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) for (int r = 0; r < 3; r++) sm[threadIdx.x * 3 + r] += sm[(threadIdx.x + o) * 3 + r];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double u = (double)(1.0f / (float)C);
        for (int k = 0; k < NC; k++) {
            part[3 * k] = k < C ? u * sm[0] : 0.0;
            part[3 * k + 1] = k < C ? u * sm[1] : 0.0;
            part[3 * k + 2] = k < C ? u * sm[2] : 0.0;
        }
    }
}

// ------------------------------------------------------------------ Maximization
// Live classes: sums over the dense posterior planes. Dormant classes (no label maps to them, prior == 0 everywhere):
// their posterior is exactly 1/C everywhere after iteration 0 (the CPU's NaN fallback) -> iteration 1 uses the
// uniform value here; from iteration 2 on they are non-zero only at the volume-border voxels (k_mstep_special).
template <int NC>
__global__ void __launch_bounds__(128, 6) k_mstep(const float *__restrict__ expec, const float *__restrict__ edorm, const float *__restrict__ prior, const float *__restrict__ T, const int *__restrict__ vox, const int *nvox, const uint8_t *__restrict__ dormFlag,
                                                 EMState *st, Geo g, int C, unsigned liveMask, int hasDormant, double *part) {
    __shared__ float scoef[kMaxBasis];
    __shared__ double sm[4 * 3 * NC];
    if (threadIdx.x < kMaxBasis) scoef[threadIdx.x] = st->coef[threadIdx.x];
    __syncthreads();
    const bool useBias = st->biasActive != 0;
    const bool it0 = st->iter == 0;
    const bool it1 = st->iter == 1;
    const float uni = 1.0f / (float)C;
    double S0[NC], S1[NC], S2[NC];
    float F0[NC], F1[NC], F2[NC];     // short float partials (<= kFlush tiles), flushed to double: cuts FP64 work 4x
#pragma unroll
    for (int c = 0; c < NC; c++) { S0[c] = 0; S1[c] = 0; S2[c] = 0; F0[c] = 0; F1[c] = 0; F2[c] = 0; }
    int na = *nvox;
    constexpr int kFlush = 4;
    int sinceFlush = 0;
    for (int base = blockIdx.x * kTileX; base < na; base += gridDim.x * kTileX) {
        int li = base + threadIdx.x;
        if (li < na) {
            int vi = vox[li];
            VP p; p.v = (size_t)vi; p.x = p.y = p.z = 0;
            float y = T[p.v];
            if (useBias) {
                p = vp_decode(vi, g.nx, g.ny);
                y = y + bias_eval(scoef, g.order, posf(p.x, g.hx, g.ihx), posf(p.y, g.hy, g.ihy), posf(p.z, g.hz, g.ihz));
            }
            float e[NC];
            if (it0) {
                // normalised prior (live planes only; dormant planes are exactly zero)
                float s = 0.f;
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    float x = 0.f;
                    if (c < C && ((liveMask >> c) & 1)) {
                        x = prior[(size_t)c * g.N + p.v];
                        if (x < 0.0f || x != x || x > 1000) x = 0.0f;
                        s += x;
                    }
                    e[c] = x;
                }
                if (s > 0 && s < 1000) {
#pragma unroll
                    for (int c = 0; c < NC; c++) e[c] = e[c] / s;
                } else {
#pragma unroll
                    for (int c = 0; c < NC; c++) e[c] = 1.0f / (float)C;
                }
            } else {
                // dormant classes: uniform in iteration 1, afterwards non-zero only at voxels that fell back to the uniform posterior
                const bool dflag = hasDormant && !it1 && dormFlag[p.v];
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    float x = 0.f;
                    if (c < C) {
                        if ((liveMask >> c) & 1) x = expec[(size_t)c * g.N + p.v];
                        else if (it1) x = uni;
                        else if (dflag) x = edorm[(size_t)c * g.N + p.v];
                    }
                    e[c] = x;
                }
            }
#pragma unroll
            for (int c = 0; c < NC; c++) {
                float cv = e[c] * y;
                if (cv == cv) {
                    F0[c] += e[c];
                    F1[c] += cv;
                    F2[c] += cv * y;
                }
            }
        }
        if (++sinceFlush == kFlush) {
            sinceFlush = 0;
#pragma unroll
            for (int c = 0; c < NC; c++) { S0[c] += (double)F0[c]; S1[c] += (double)F1[c]; S2[c] += (double)F2[c]; F0[c] = 0; F1[c] = 0; F2[c] = 0; }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; c++) { S0[c] += (double)F0[c]; S1[c] += (double)F1[c]; S2[c] += (double)F2[c]; }
    double val[3 * NC];
#pragma unroll
    for (int c = 0; c < NC; c++) { val[3 * c] = S0[c]; val[3 * c + 1] = S1[c]; val[3 * c + 2] = S2[c]; }
    block_reduce_store<3 * NC>(val, sm, part);
}

// dormant-class sums over the special (border) voxels, iteration >= 2
template <int NC>
__global__ void __launch_bounds__(128) k_mstep_special(const float *expec, const float *T, const int *spec, const int *nspec, const EMState *st, Geo g, int C,
                                                      unsigned liveMask, double *part) {
    __shared__ float scoef[kMaxBasis];
    __shared__ double sm[4 * 3 * NC];
    if (threadIdx.x < kMaxBasis) scoef[threadIdx.x] = st->coef[threadIdx.x];
    __syncthreads();
    const bool useBias = st->biasActive != 0;
    double S0[NC], S1[NC], S2[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) { S0[c] = 0; S1[c] = 0; S2[c] = 0; }
    if (st->iter >= 2) {
        int n = *nspec;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
            size_t v = (size_t)spec[i];
            int x = (int)(v % g.nx), y = (int)((v / g.nx) % g.ny), z = (int)(v / ((size_t)g.nx * g.ny));
            float yy = T[v];
            if (useBias) yy = yy + bias_eval(scoef, g.order, posf(x, g.hx, g.ihx), posf(y, g.hy, g.ihy), posf(z, g.hz, g.ihz));
#pragma unroll
            for (int c = 0; c < NC; c++) {
                if (c < C && !((liveMask >> c) & 1)) {
                    float e = expec[(size_t)c * g.N + v];
                    float cv = e * yy;
                    if (cv == cv) { S0[c] += (double)e; S1[c] += (double)cv; S2[c] += (double)cv * (double)yy; }
                }
            }
        }
    }
    double val[3 * NC];
#pragma unroll
    for (int c = 0; c < NC; c++) { val[3 * c] = S0[c]; val[3 * c + 1] = S1[c]; val[3 * c + 2] = S2[c]; }
    block_reduce_store<3 * NC>(val, sm, part);
}

template <int NC>
__global__ void k_mfinal(const double *part, int nblocks, const double *partS, int nblocksS, EMState *st, int C) {
    constexpr int R = 3 * NC;
    constexpr int J = 256 / R;                // partial-sum lanes per output
    __shared__ double lane[J][R];
    __shared__ double sum[R];
    int r = threadIdx.x % R, j = threadIdx.x / R;
    if (j < J) {
        double s = 0;
#pragma unroll 4
        for (int b = j; b < nblocks; b += J) s += part[(size_t)b * R + r];
        for (int b = j; b < nblocksS; b += J) s += partS[(size_t)b * R + r];
        lane[j][r] = s;
    }
    __syncthreads();
    if (threadIdx.x < R) {
        double s = 0;
        for (int k = 0; k < J; k++) s += lane[k][threadIdx.x];      // fixed order
        sum[threadIdx.x] = s;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int c = 0; c < C; c++) {
            double S0 = sum[3 * c], S1 = sum[3 * c + 1], S2 = sum[3 * c + 2];
            float sp = (float)S0;
            if (sp == sp && sp > 0) {
                float Mn = (float)(S1 / S0);
                st->M[c] = Mn;
                double ts2 = S2 - 2.0 * (double)Mn * S1 + (double)Mn * (double)Mn * S0;
                float vv = (float)(ts2 / S0);
                if (vv > 0 && !isnan(vv)) st->V[c] = vv;
            }
            float Vc = st->V[c];
            st->inv_sqrt[c] = 1.0f / sqrtf((float)(2.0 * M_PI * (double)Vc));
            st->inv_v[c] = 1.0f / Vc;
        }
    }
}

// ------------------------------------------------------------------ Expectation
// E-step fused with the MRF update.  The MRF field of iteration i-1 (CPU: RunMRF3D after the E-step) is a pure function of the
// posterior of iteration i-1 (read from eIn: 6 neighbours, first-voxel aliasing outside the mask) and of the prior, so it is evaluated
// here, on the fly, instead of being stored and re-read.  Dormant classes (prior == 0) have MRF == 0 and a common neighbour-mass factor
// that cancels in the normalisation -> live classes only.
template <int NC>
__global__ void __launch_bounds__(128, 5) k_estep(float *__restrict__ eOut, const float *__restrict__ eIn, float *__restrict__ eDorm, const float *__restrict__ prior,
                                                 const float *__restrict__ T, const uint8_t *__restrict__ mask, const int *__restrict__ vox, const int *nvox,
                                                 uint8_t *__restrict__ dormFlag, EMState *st, Geo g, int C, unsigned liveMask, int hasDormant, int shortcut,
                                                 float beta, double *part) {
    __shared__ float scoef[kMaxBasis];
    __shared__ double sm[4];
    __shared__ float sM[kMaxClasses], sIV[kMaxClasses], sIS[kMaxClasses];
    if (threadIdx.x < kMaxBasis) scoef[threadIdx.x] = st->coef[threadIdx.x];
    if (threadIdx.x < kMaxClasses) { sM[threadIdx.x] = st->M[threadIdx.x]; sIV[threadIdx.x] = st->inv_v[threadIdx.x]; sIS[threadIdx.x] = st->inv_sqrt[threadIdx.x]; }
    __syncthreads();
    const bool useBias = st->biasActive != 0;
    const bool it0 = st->iter == 0;
    const bool it1 = st->iter == 1;
    // iteration 1 with dormant classes (shortcut): the MRF field of iteration 0 equals the normalised prior (uniform neighbours)
    const bool priorAsMrf = shortcut && it1;
    const bool needNb = !it0 && !priorAsMrf;
    const size_t first = (size_t)st->firstMasked;
    float efirst[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) efirst[c] = (needNb && c < C && ((liveMask >> c) & 1)) ? eIn[(size_t)c * g.N + first] : 0.f;
    double ll = 0;
    int na = *nvox;
    const size_t plane = (size_t)g.nx * g.ny;
    for (int base = blockIdx.x * kTileX; base < na; base += gridDim.x * kTileX) {
        int li = base + threadIdx.x;
        if (li >= na) continue;
        VP p = vp_decode(vox[li], g.nx, g.ny);
        // MRF field is only valid at voxels updated by RunMRF3D (interior of the volume)
        bool interior = p.x >= 1 && p.x <= g.nx - 2 && p.y >= 1 && p.y <= g.ny - 2 && p.z >= 1 && p.z <= g.nz - 2;
        if (hasDormant && !interior) continue;                       // handled by k_estep_special
        if (hasDormant && it0) {                                     // NaN fallback of the CPU: uniform posterior, no loglik term
            float u = (float)(1) / (float)(C);
#pragma unroll
            for (int c = 0; c < NC; c++) if (c < C && ((liveMask >> c) & 1)) eOut[(size_t)c * g.N + p.v] = u;
            continue;
        }
        float y = T[p.v];
        if (useBias) y = y + bias_eval(scoef, g.order, posf(p.x, g.hx, g.ihx), posf(p.y, g.hy, g.ihy), posf(p.z, g.hz, g.ihz));
        const bool mrfValid = !it0 && interior;
        // ---- MRF prior of this voxel
        float pr[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) pr[c] = 1.0f;
        if (mrfValid) {
            float prn[NC];
            float ps = 0.f;
#pragma unroll
            for (int c = 0; c < NC; c++) {
                float x = 0.f;
                if (c < C && ((liveMask >> c) & 1)) {
                    x = prior[(size_t)c * g.N + p.v];
                    if (x < 0.0f || x != x || x > 1000) x = 0.0f;
                    ps += x;
                }
                prn[c] = x;
            }
            if (ps > 0 && ps < 1000) {
#pragma unroll
                for (int c = 0; c < NC; c++) prn[c] = prn[c] / ps;
            } else {
#pragma unroll
                for (int c = 0; c < NC; c++) prn[c] = (c < C && ((liveMask >> c) & 1)) ? 1.0f / (float)C : 0.f;
            }
            if (priorAsMrf) {
#pragma unroll
                for (int c = 0; c < NC; c++) pr[c] = prn[c];
            } else {
                // neighbour order of RunMRF3D: West=y-1, East=y+1, North=x-1, South=x+1 (G), Top=z-1, Bottom=z+1 (H)
                size_t nbi[6] = {p.v - g.nx, p.v + g.nx, p.v - 1, p.v + 1, p.v - plane, p.v + plane};
                bool nm[6];
#pragma unroll
                for (int k = 0; k < 6; k++) nm[k] = mask[nbi[k]] != 0;
                float G[NC], H[NC];
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    G[c] = 0.f; H[c] = 0.f;
                    if (c < C && ((liveMask >> c) & 1)) {
                        float ev[6];
#pragma unroll
                        for (int k = 0; k < 6; k++) ev[k] = eIn[(size_t)c * g.N + nbi[k]];
                        float gg = 0.f, hh = 0.f;
                        gg += nm[0] ? ev[0] : efirst[c];
                        gg += nm[1] ? ev[1] : efirst[c];
                        gg += nm[2] ? ev[2] : efirst[c];
                        gg += nm[3] ? ev[3] : efirst[c];
                        hh += nm[4] ? ev[4] : efirst[c];
                        hh += nm[5] ? ev[5] : efirst[c];
                        G[c] = gg; H[c] = hh;
                    }
                }
                float tmp[NC];
                float msum = 0.f;
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    float tc = 0.f;
                    if (c < C && ((liveMask >> c) & 1)) {
#pragma unroll
                        for (int n = 0; n < NC; n++) {
                            if (n < C && n != c && ((liveMask >> n) & 1)) tc = __fsub_rn(tc, __fadd_rn(__fmul_rn(beta, G[n]), __fmul_rn(beta, H[n])));
                        }
                        tc = __fmul_rn(expf(tc), prn[c]);
                        msum = __fadd_rn(msum, tc);
                    }
                    tmp[c] = tc;
                }
#pragma unroll
                for (int c = 0; c < NC; c++) pr[c] = tmp[c] / msum;
            }
        }
        float e[NC];
        float sum = 0.f;
#pragma unroll
        for (int c = 0; c < NC; c++) {
            if (c < C && ((liveMask >> c) & 1)) {
                float diff = y - sM[c];
                float mahal = __fsub_rn(0.0f, __fmul_rn(__fmul_rn(__fmul_rn(0.5f, diff), sIV[c]), diff));
                e[c] = __fmul_rn(pr[c], __fmul_rn(sIS[c], expf(mahal)));
                sum = __fadd_rn(sum, e[c]);
            } else e[c] = 0.f;
        }
        if (sum <= 0.0 || sum != sum) {
            // CPU fallback: uniform posterior for ALL classes (the dormant ones too -> keep them in their planes and flag the voxel)
            float u = (float)(1) / (float)(C);
#pragma unroll
            for (int c = 0; c < NC; c++) {
                if (c < C && ((liveMask >> c) & 1)) eOut[(size_t)c * g.N + p.v] = u;
                else if (c < C && hasDormant) eDorm[(size_t)c * g.N + p.v] = u;
            }
            if (hasDormant) dormFlag[p.v] = 1;
        } else {
#pragma unroll
            for (int c = 0; c < NC; c++) if (c < C && ((liveMask >> c) & 1)) eOut[(size_t)c * g.N + p.v] = e[c] / sum;
            if (hasDormant && dormFlag[p.v]) {
#pragma unroll
                for (int c = 0; c < NC; c++) if (c < C && !((liveMask >> c) & 1)) eDorm[(size_t)c * g.N + p.v] = 0.f;
                dormFlag[p.v] = 0;
            }
            ll += (double)logf(sum);
        }
    }
    double val[1] = {ll};
    block_reduce_store<1>(val, sm, part);
}

// border voxels (not touched by the MRF update): all classes, prior = 1, exactly the CPU expression; the dormant classes
// keep their posterior in their (otherwise zero) planes of the posterior buffer.
template <int NC>
__global__ void __launch_bounds__(128) k_estep_special(float *eOut, float *eDorm, const float *T, const int *spec, const int *nspec, EMState *st, Geo g, int C, unsigned liveMask, double *part) {
    __shared__ float scoef[kMaxBasis];
    __shared__ double sm[4];
    __shared__ float sM[kMaxClasses], sIV[kMaxClasses], sIS[kMaxClasses];
    if (threadIdx.x < kMaxBasis) scoef[threadIdx.x] = st->coef[threadIdx.x];
    if (threadIdx.x < kMaxClasses) { sM[threadIdx.x] = st->M[threadIdx.x]; sIV[threadIdx.x] = st->inv_v[threadIdx.x]; sIS[threadIdx.x] = st->inv_sqrt[threadIdx.x]; }
    __syncthreads();
    const bool useBias = st->biasActive != 0;
    const bool it0 = st->iter == 0;
    double ll = 0;
    int n = *nspec;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        size_t v = (size_t)spec[i];
        int x = (int)(v % g.nx), y = (int)((v / g.nx) % g.ny), z = (int)(v / ((size_t)g.nx * g.ny));
        if (it0) {
            float u = (float)(1) / (float)(C);
#pragma unroll
            for (int c = 0; c < NC; c++) if (c < C) ((((liveMask >> c) & 1) ? eOut : eDorm))[(size_t)c * g.N + v] = u;
            continue;
        }
        float yy = T[v];
        if (useBias) yy = yy + bias_eval(scoef, g.order, posf(x, g.hx, g.ihx), posf(y, g.hy, g.ihy), posf(z, g.hz, g.ihz));
        float e[NC];
        float sum = 0.f;
#pragma unroll
        for (int c = 0; c < NC; c++) {
            if (c < C) {
                float diff = yy - sM[c];
                float mahal = __fsub_rn(0.0f, __fmul_rn(__fmul_rn(__fmul_rn(0.5f, diff), sIV[c]), diff));
                e[c] = __fmul_rn(1.0f, __fmul_rn(sIS[c], expf(mahal)));
                sum = __fadd_rn(sum, e[c]);
            } else e[c] = 0.f;
        }
        if (sum <= 0.0 || sum != sum) {
            float u = (float)(1) / (float)(C);
#pragma unroll
            for (int c = 0; c < NC; c++) if (c < C) ((((liveMask >> c) & 1) ? eOut : eDorm))[(size_t)c * g.N + v] = u;
        } else {
#pragma unroll
            for (int c = 0; c < NC; c++) if (c < C) ((((liveMask >> c) & 1) ? eOut : eDorm))[(size_t)c * g.N + v] = e[c] / sum;
            ll += (double)logf(sum);
        }
    }
    double val[1] = {ll};
    block_reduce_store<1>(val, sm, part);
}

__global__ void k_efinal(const double *part, int nblocks, const double *partS, int nblocksS, EMState *st) {
    __shared__ double sm[256];
    double s = 0;
    for (int b = threadIdx.x; b < nblocks; b += blockDim.x) s += part[b];
    for (int b = threadIdx.x; b < nblocksS; b += blockDim.x) s += partS[b];
    sm[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {          // fixed tree
        if (threadIdx.x < o) sm[threadIdx.x] += sm[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) st->loglik = sm[0];   // (the CPU accumulates this in float per OpenMP thread; here: double, fixed order)
}

// ------------------------------------------------------------------ Bias field
// device-side gate: RunBiasField's condition
__device__ __forceinline__ bool bias_gate(const EMState *st, float ratio) {
    return (((st->loglik - st->oldloglik) / fabs(st->oldloglik)) < (double)ratio && st->iter > 3) || ratio == 0.0f;
}

// One warp per line (iy, iz both even): power sums of W and W*(y - mean) along x over every second voxel.
// line record: [0..8] sum W x^p, [9..13] sum Wr x^p
template <int NC>
__global__ void k_bias_lines(const float *expec, const float *edorm, const float *T, const uint8_t *mask, const uint8_t *dormFlag, const EMState *st, Geo g, int C, unsigned liveMask, int hasDormant, float ratio,
                             double *lineAcc, int nly, int nlz) {
    if (!bias_gate(st, ratio)) return;
    int line = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (line >= nly * nlz) return;
    int iy = 2 * (line % nly), iz = 2 * (line / nly);
    float invV[NC], cM[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) { invV[c] = st->inv_v[c]; cM[c] = st->M[c]; }
    float sw[9], sr[5];      // per-lane partial sums over at most nx/64 samples in float; the cross-lane / cross-line sums are double
#pragma unroll
    for (int k = 0; k < 9; k++) sw[k] = 0;
#pragma unroll
    for (int k = 0; k < 5; k++) sr[k] = 0;
    size_t rowbase = (size_t)g.nx * (iy + (size_t)g.ny * iz);
    for (int ix = 2 * lane; ix < g.nx; ix += 64) {
        size_t v = rowbase + ix;
        if (!mask[v]) continue;
        bool border = !(ix >= 1 && ix <= g.nx - 2 && iy >= 1 && iy <= g.ny - 2 && iz >= 1 && iz <= g.nz - 2);
        if (hasDormant && !border) border = dormFlag[v] != 0;   // dormant classes are non-zero at border voxels and at fallback voxels
        float Wtmp = 0.f, Wi = 0.f, Yest = 0.f, Ysum = 0.f;
#pragma unroll
        for (int c = 0; c < NC; c++) {
            if (c < C && (((liveMask >> c) & 1) || (hasDormant && border))) {
                float e = (((liveMask >> c) & 1) ? expec : edorm)[(size_t)c * g.N + v];
                Wtmp += e * invV[c];
                float Wij = e * invV[c];
                Wi += Wij;
                Yest += Wij * cM[c];
                Ysum += Wij;
            }
        }
        float wr = Wi * (T[v] - (Yest / Ysum));
        float xp = posf(ix, g.hx, g.ihx);
        float pw = 1.0f;
#pragma unroll
        for (int k = 0; k < 9; k++) {
            sw[k] += Wtmp * pw;
            if (k < 5) sr[k] += wr * pw;
            pw *= xp;
        }
    }
#pragma unroll
    for (int k = 0; k < 9; k++) {
        double x = (double)sw[k];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
        if (lane == 0) lineAcc[(size_t)line * 14 + k] = x;
    }
#pragma unroll
    for (int k = 0; k < 5; k++) {
        double x = (double)sr[k];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
        if (lane == 0) lineAcc[(size_t)line * 14 + 9 + k] = x;
    }
}

// moments: thread t < nMom: sum over lines S_p(line) * y^q z^r ; thread nMom+k: WR_k.  y^q and z^r of the block's lines come from
// shared tables (double).
__global__ void k_bias_moments(const double *lineAcc, const EMState *st, Geo g, float ratio, int nly, int nlz, int nMom, int nBas, double *momAcc) {
    if (!bias_gate(st, ratio)) return;
    __shared__ double ypw[kLinesPerBlock][9], zpw[kLinesPerBlock][9], la[kLinesPerBlock][14];
    int t = threadIdx.x;
    int nlines = nly * nlz;
    int l0 = blockIdx.x * kLinesPerBlock;
    int nl = min(kLinesPerBlock, nlines - l0);
    if (t < kLinesPerBlock * 2 && t / 2 < nl) {
        int li = t / 2, l = l0 + li;
        bool isy = (t & 1) == 0;
        double pos = isy ? (double)posf(2 * (l % nly), g.hy, g.ihy) : (double)posf(2 * (l / nly), g.hz, g.ihz);
        double pw = 1.0;
        for (int k = 0; k < 9; k++) { (isy ? ypw : zpw)[li][k] = pw; pw *= pos; }
    }
    for (int i = t; i < nl * 14; i += blockDim.x) la[i / 14][i % 14] = lineAcc[(size_t)(l0 + i / 14) * 14 + i % 14];
    __syncthreads();
    double acc = 0;
    if (t < nMom + nBas) {
        int p, q, r, which;
        if (t < nMom) { p = c_mom[t][0]; q = c_mom[t][1]; r = c_mom[t][2]; which = 0; }
        else { int k = t - nMom; p = c_bas[k][0]; q = c_bas[k][1]; r = c_bas[k][2]; which = 9; }
        for (int li = 0; li < nl; li++) acc += la[li][which + p] * ypw[li][q] * zpw[li][r];
    }
    if (t < 256) momAcc[(size_t)blockIdx.x * 256 + t] = acc;
}

// single block: reduce moment partials, assemble A'WA and A'Wr, solve (Gauss-Jordan, partial pivoting, double)
__global__ void __launch_bounds__(1024) k_bias_solve(const double *momAcc, int nblk, EMState *st, float ratio, int nMom, int nBas) {
    __shared__ double mom[256];
    __shared__ double lanes[4][256];
    __shared__ double A[kMaxBasis][kMaxBasis + 1];
    __shared__ int piv;
    if (!bias_gate(st, ratio)) { if (threadIdx.x == 0) st->biasRan = 0; return; }
    int t = threadIdx.x;
    {   // 4 lanes per moment, fixed order
        double s = 0;
#pragma unroll 8
        for (int b = t / 256; b < nblk; b += 4) s += momAcc[(size_t)b * 256 + (t & 255)];
        lanes[t / 256][t & 255] = s;
    }
    __syncthreads();
    if (t < 256) mom[t] = ((lanes[0][t] + lanes[1][t]) + lanes[2][t]) + lanes[3][t];
    __syncthreads();
    // assemble (every thread fills some entries)
    for (int e = t; e < nBas * nBas; e += blockDim.x) {
        int i = e / nBas, j = e % nBas;
        A[i][j] = mom[c_midx[c_bas[i][0] + c_bas[j][0]][c_bas[i][1] + c_bas[j][1]][c_bas[i][2] + c_bas[j][2]]];
    }
    if (t < nBas) {
        double wr = mom[nMom + t];
        if (wr != wr) wr = 1.0;          // CPU: "if nan, set to 1"
        A[t][nBas] = wr;
    }
    __syncthreads();
    for (int k = 0; k < nBas; k++) {
        if (t == 0) {
            int best = k; double bv = fabs(A[k][k]);
            for (int i = k + 1; i < nBas; i++) if (fabs(A[i][k]) > bv) { bv = fabs(A[i][k]); best = i; }
            piv = best;
        }
        __syncthreads();
        if (piv != k) {
            for (int j = t; j <= nBas; j += blockDim.x) { double tmp = A[k][j]; A[k][j] = A[piv][j]; A[piv][j] = tmp; }
        }
        __syncthreads();
        double pv = A[k][k];
        __syncthreads();
        if (t <= nBas) A[k][t] = A[k][t] / pv;
        __syncthreads();
        if (t < nBas && t != k) {
            double f = A[t][k];
            for (int j = k; j <= nBas; j++) A[t][j] -= f * A[k][j];
        }
        __syncthreads();
    }
    if (t < nBas) st->coef[t] = (float)A[t][nBas];
    if (t == 0) { st->biasActive = 1; st->biasRan = 1; }
}

// ------------------------------------------------------------------ iteration bookkeeping
__global__ void k_iter_end(EMState *st) {
    if (threadIdx.x == 0) { st->oldloglik = st->loglik; st->iter++; }
}

// ------------------------------------------------------------------ bias corrected image (seg_EM::GetBiasCorrected)
__global__ void k_biascorr(const float *target, const float *bmask, float *out, const EMState *st, Geo g) {
    __shared__ float scoef[kMaxBasis];
    if (threadIdx.x < kMaxBasis) scoef[threadIdx.x] = st->coef[threadIdx.x];
    __syncthreads();
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y, z = blockIdx.z;
    if (x >= g.nx) return;
    size_t v = (size_t)x + (size_t)g.nx * (y + (size_t)g.ny * z);
    float rmin = st->rmin, rmax = st->rmax;
    float xv = target[v];
    // normalised image (seg_EM::InitializeAndNormalizeImage), recomputed on the fly for every voxel
    float T = (float)((double)logf((((xv) - rmin) / (rmax - rmin)) + 1) / 0.693147181);
    float bm = bmask[v];
    float b = 0.f;
    if (bm != 0.f) {
        b = bias_eval(scoef, g.order, posf(x, g.hx, g.ihx), posf(y, g.hy, g.ihy), posf(z, g.hz, g.ihz));
        b = __fmul_rn(b, bm);
    }
    // exp((b+T)*ln2) - 1 in float (the CPU uses a double exp; the difference is below one float ulp of the result)
    float to_resize = expf(__fmul_rn(__fadd_rn(b, T), 0.693147181f)) - 1.0f;
    out[v] = __fadd_rn(__fmul_rn(to_resize, __fsub_rn(rmax, rmin)), rmin);
}

// ------------------------------------------------------------------ host orchestration
static void init_tables(int order) {
    static bool done = false;
    static int cur = -1;
    if (done && cur == order) return;
    uint8_t mom[kMaxMoments][3] = {{0}}, bas[kMaxBasis][3] = {{0}}, midx[9][9][9];
    memset(midx, 0, sizeof(midx));
    int k = 0;
    for (int tot = 0; tot <= 2 * kMaxOrder; tot++)
        for (int p = 0; p <= tot; p++)
            for (int q = 0; q <= tot - p; q++) {
                int r = tot - p - q;
                mom[k][0] = p; mom[k][1] = q; mom[k][2] = r; midx[p][q][r] = k; k++;
            }
    k = 0;
    for (int o = 0; o <= kMaxOrder; o++)
        for (int xo = 0; xo <= o; xo++)
            for (int yo = 0; yo <= o - xo; yo++) { bas[k][0] = xo; bas[k][1] = yo; bas[k][2] = o - yo - xo; k++; }
    CUDA_CHECK(cudaMemcpyToSymbol(c_mom, mom, sizeof(mom)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_bas, bas, sizeof(bas)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_midx, midx, sizeof(midx)));
    done = true; cur = order;
}

struct IsSpecial {
    const uint8_t *mask; int nx, ny, nz;
    __device__ bool operator()(const int &i) const {
        if (!mask[i]) return false;
        int x = i % nx, y = (i / nx) % ny, z = i / (nx * ny);
        return !(x >= 1 && x <= nx - 2 && y >= 1 && y <= ny - 2 && z >= 1 && z <= nz - 2);
    }
};
__global__ void k_clear_spec(float *e0, float *e1, const int *spec, const int *nspec, size_t N, int C, unsigned liveMask) {
    int n = *nspec;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        for (int c = 0; c < C; c++)
            if (!((liveMask >> c) & 1)) { e0[(size_t)c * N + spec[i]] = 0.f; e1[(size_t)c * N + spec[i]] = 0.f; }
}

// Restore the "zero outside the mask" state of the two posterior buffers by clearing the previous run's masked voxels only
// (the buffers are zero-initialised at allocation and the E-step writes masked voxels only).
__global__ void k_clear_prev(float *e0, float *e1, uint8_t *flag, const int *vox, const int *nvox, size_t N, int C, unsigned liveMask) {
    int n = *nvox;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        size_t v = (size_t)vox[i];
        for (int c = 0; c < C; c++)
            if ((liveMask >> c) & 1) { e0[(size_t)c * N + v] = 0.f; e1[(size_t)c * N + v] = 0.f; }
        if (flag[v]) {
            for (int c = 0; c < C; c++)
                if (!((liveMask >> c) & 1)) { e0[(size_t)c * N + v] = 0.f; e1[(size_t)c * N + v] = 0.f; }
            flag[v] = 0;
        }
    }
}

// The final posterior lives in one of the two buffers; its dormant planes must hold the dormant posterior of the border /
// fallback voxels (kept in buffer 0 during the iterations).
__global__ void k_copy_dormant(float *dst, const float *src, const int *vox, const int *nvox, const uint8_t *flag, int nx, int ny, int nz, size_t N, int C, unsigned liveMask) {
    int n = *nvox;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        int v = vox[i];
        int x = v % nx, y = (v / nx) % ny, z = v / (nx * ny);
        bool border = !(x >= 1 && x <= nx - 2 && y >= 1 && y <= ny - 2 && z >= 1 && z <= nz - 2);
        if (border || flag[v])
            for (int c = 0; c < C; c++)
                if (!((liveMask >> c) & 1)) dst[(size_t)c * N + v] = src[(size_t)c * N + v];
    }
}

template <int NC>
static void em_run(EmContext::Impl &I, const float *target) {
    cudaStream_t s = I.stream;
    Geo g = make_geo(I);
    const int C = I.C;
    const unsigned liveMask = I.liveMask;
    const int hasDormant = I.hasDormant;
    int order = I.prm.bias_order;
    int nBas = (order + 1) * (order + 2) * (order + 3) / 6;
    int nMom = 0;
    for (int tot = 0; tot <= 2 * order; tot++) nMom += (tot + 1) * (tot + 2) / 2;
    init_tables(order);
    // posterior buffers: back to all-zero (previous run's masked voxels; dormant planes also at the border voxels)
    k_clear_prev<<<kEmGrid, 256, 0, s>>>(I.expec, I.expec2, I.dormFlag, I.voxList, I.numActive, I.d.N, C, liveMask);
    if (hasDormant) k_clear_spec<<<64, 128, 0, s>>>(I.expec, I.expec2, I.specList, I.numSpec, I.d.N, C, liveMask);
    // masked-voxel list (sorted), statistics, normalised image
    {
        thrust::counting_iterator<int> it(0);
        IsMasked pm{I.emMask};
        if (!I.cubTmp) {
            size_t q = 0;
            CUDA_CHECK(cub::DeviceSelect::If(nullptr, q, it, I.voxList, I.numActive, (int)I.d.N, pm, s));
            I.cubTmpBytes = q;
            I.cubTmp = I.dalloc<uint8_t>(q + 16);
        }
        size_t bytes = I.cubTmpBytes;
        CUDA_CHECK(cub::DeviceSelect::If(I.cubTmp, bytes, it, I.voxList, I.numActive, (int)I.d.N, pm, s));
    }
    k_list_stats<<<kEmGrid, kTileX, 0, s>>>(I.voxList, I.numActive, target, I.pmin, I.pmax);
    k_list_final<<<1, 256, 0, s>>>(I.voxList, I.numActive, I.pmin, I.pmax, kEmGrid, I.st);
    if (hasDormant) {
        IsBorder pb{g.nx, g.ny, g.nz};
        if (!I.cubTmp2) {
            size_t q = 0;
            CUDA_CHECK(cub::DeviceSelect::If(nullptr, q, I.voxList, I.specList, I.numSpec, (int)I.d.N, pb, s));
            I.cubTmpBytes2 = q;
            I.cubTmp2 = I.dalloc<uint8_t>(q + 16);
        }
        // select over the list: the number of list entries is only known on the device, so use the upper bound N and let the
        // device-side count (num_items is host side) -- see below: we first need the count on the host.
    }
    k_normalize<<<kEmGrid, 256, 0, s>>>(target, I.T, I.voxList, I.numActive, I.st, I.pT);
    int nact = 0, nspec = 0;
    CUDA_CHECK(cudaMemcpyAsync(&nact, I.numActive, sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    if (hasDormant) {
        IsBorder pb{g.nx, g.ny, g.nz};
        size_t bytes2 = I.cubTmpBytes2;
        CUDA_CHECK(cub::DeviceSelect::If(I.cubTmp2, bytes2, I.voxList, I.specList, I.numSpec, nact, pb, s));
        CUDA_CHECK(cudaMemcpyAsync(&nspec, I.numSpec, sizeof(int), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
    }
    I.nActHost = nact;
    I.nSpecHost = nspec;
    cudaEventRecord(I.ev[3], s);   // end of em_init
    int nsm = 0, dev = 0;
    cudaGetDevice(&dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    int perM = 0, perE = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perM, k_mstep<NC>, kTileX, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perE, k_estep<NC>, kTileX, 0);
    int need = (nact + kTileX - 1) / kTileX;
    int grid = std::min(std::min(need, nsm * std::max(perM, 1)), kEmGrid);      // one persistent wave per kernel
    int gridE = std::min(std::min(need, nsm * std::max(perE, 1)), kEmGrid);
    if (grid < 1) grid = 1;
    if (gridE < 1) gridE = 1;
    int gridS = hasDormant ? (nspec + 127) / 128 : 0;
    if (gridS > kSpecGrid) gridS = kSpecGrid;
    if (hasDormant && gridS < 1) gridS = 1;
    int nly = (I.d.ny + 1) / 2, nlz = (I.d.nz + 1) / 2;
    int nlines = nly * nlz;
    int linesBlocks = (nlines + 7) / 8;
    int momBlocks = (nlines + kLinesPerBlock - 1) / kLinesPerBlock;
    I.iters = 0;
    const bool shortcut = hasDormant && I.prm.max_iter >= 2;     // iterations 0/1 in closed form (closed form, see comment above)
    float *buf[2] = {I.expec, I.expec2};                          // posterior of iteration i lives in buf[i & 1]; dormant planes in buf[0]
    int last = -1;
    for (int it = 0; it < I.prm.max_iter + 1; it++) {
        float *eIn = buf[(it + 1) & 1], *eOut = buf[it & 1];
        if (shortcut && it == 1) {
            k_m_uniform<NC><<<1, 256, 0, s>>>(I.pT, kEmGrid, C, I.red);
            k_mfinal<NC><<<1, 256, 0, s>>>(I.red, 1, I.redS, 0, I.st, C);
        } else {
            k_mstep<NC><<<grid, kTileX, 0, s>>>(eIn, buf[0], I.prior, I.T, I.voxList, I.numActive, I.dormFlag, I.st, g, C, liveMask, hasDormant, I.red);
            if (hasDormant) k_mstep_special<NC><<<gridS, kTileX, 0, s>>>(buf[0], I.T, I.specList, I.numSpec, I.st, g, C, liveMask, I.redS);
            k_mfinal<NC><<<1, 256, 0, s>>>(I.red, grid, I.redS, gridS, I.st, C);
        }
        if (shortcut && it == 0) {
            // E-step / MRF of iteration 0 are the uniform fallback: nothing to compute, loglik = 0
            k_efinal<<<1, 256, 0, s>>>(I.red, 0, I.redSL, 0, I.st);
        } else {
            k_estep<NC><<<gridE, kTileX, 0, s>>>(eOut, eIn, buf[0], I.prior, I.T, I.emMask, I.voxList, I.numActive, I.dormFlag, I.st, g, C, liveMask, hasDormant,
                                                 shortcut ? 1 : 0, I.prm.mrf_beta, I.red);
            if (hasDormant) k_estep_special<NC><<<gridS, kTileX, 0, s>>>(eOut, buf[0], I.T, I.specList, I.numSpec, I.st, g, C, liveMask, I.redSL);
            k_efinal<<<1, 256, 0, s>>>(I.red, gridE, I.redSL, gridS, I.st);
            k_bias_lines<NC><<<linesBlocks, 256, 0, s>>>(eOut, buf[0], I.T, I.emMask, I.dormFlag, I.st, g, C, liveMask, hasDormant, I.prm.bias_ratio, I.lineAcc, nly, nlz);
            k_bias_moments<<<momBlocks, 256, 0, s>>>(I.lineAcc, I.st, g, I.prm.bias_ratio, nly, nlz, nMom, nBas, I.momAcc);
            k_bias_solve<<<1, 1024, 0, s>>>(I.momAcc, momBlocks, I.st, I.prm.bias_ratio, nMom, nBas);
        }
        CUDA_CHECK(cudaMemcpyAsync(I.hostSt, I.st, sizeof(EMState), cudaMemcpyDeviceToHost, s));
        k_iter_end<<<1, 1, 0, s>>>(I.st);
        CUDA_CHECK(cudaStreamSynchronize(s));
        I.iters = it + 1;
        const EMState &h = *I.hostSt;
        if (getenv("SEGEM_VERBOSE")) {
            if (it == 0) printf("masked %d, nspec %d\n", h.numMasked, nspec);
            printf("it %d loglik %.7g old %.7g biasRan %d\n", h.iter, h.loglik, h.oldloglik, h.biasRan);
            for (int c = 0; c < C; c++) printf("  M[%d]=%.8f V[%d]=%.8f\n", c, h.M[c], c, h.V[c]);
        }
        double loglik = h.loglik, old = h.oldloglik;
        int iter = h.iter;
        bool stop = ((((loglik - old) / (fabs(old + loglik) / 2.0f)) < (double)(float)I.prm.conv_crit) && iter > I.prm.min_iter) ||
                    iter >= I.prm.max_iter || (std::isinf(loglik) && iter > 3);
        if (stop) break;
    }
    last = I.iters - 1;
    I.eFinal = buf[last & 1];
    if (hasDormant && I.eFinal != buf[0])
        k_copy_dormant<<<kEmGrid, 256, 0, s>>>(I.eFinal, buf[0], I.voxList, I.numActive, I.dormFlag, g.nx, g.ny, g.nz, I.d.N, C, liveMask);
    CUDA_CHECK(cudaGetLastError());
}

void launch_em(EmContext::Impl &I, const float *target) {
    if (I.C <= 8) em_run<8>(I, target);
    else em_run<12>(I, target);
}

void launch_bias_corrected(EmContext::Impl &I) {
    cudaStream_t s = I.stream;
    Geo g = make_geo(I);
    launch_smooth_mask(I);
    dim3 grid((g.nx + 127) / 128, g.ny, g.nz);
    k_biascorr<<<grid, 128, 0, s>>>(I.targetPtr, I.wsF0, I.bc, I.st, g);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace segem
