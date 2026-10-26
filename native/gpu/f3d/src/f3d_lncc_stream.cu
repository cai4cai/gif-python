// SPDX-License-Identifier: BSD-3-Clause
// Fused, streaming LNCC statistics.
//   * channel creation (m, r, r^2, w, w^2, r*w) from the raw reference / warped images,
//   * separable Gaussian passes: x and y through shared memory, z through a ring buffer kept in registers,
//   * epilogue: local NCC value reduction and/or store of the final 6 smoothed channels.
// Arithmetic: EXACT emulates NiftyReg (float products accumulated in double, rounded to float after every pass) with an
// error-free float two-sum; !EXACT accumulates with plain float FMAs.
#include "f3d_internal.h"
#include "f3d_lncc_common.cuh"
#include <cmath>
#include <cstring>

namespace f3d {

#define TX 16
#define TY 16
#define NTHR (TX * TY)

template <bool EXACT>
__device__ __forceinline__ void acc_tap(float &hi, float &lo, float w, float v)
{
    if (EXACT) {
        float pr = __fmul_rn(w, v);
        float s = hi + pr;
        float bp = s - hi;
        lo += (hi - (s - bp)) + (pr - bp);
        hi = s;
    } else {
        hi = __fmaf_rn(w, v, hi);
    }
}

template <int S, bool EXACT, bool STORE, int CHG>
__global__ void __launch_bounds__(NTHR, 2)
    lncc_stream_k(const float *__restrict__ ref, const int *__restrict__ mask, const float *__restrict__ warped, size_t N, int nx,
                  int ny, int nz, int rx, int ry, const float *__restrict__ tapx, const float *__restrict__ tapy,
                  const float *__restrict__ tapz, double *partial, int nblk, float *stats, const int *act, int tilesX)
{
    constexpr int RZ = (S - 1) / 2;
    constexpr int NC = (CHG == 0) ? 6 : 3;     // channels handled by this kernel
    constexpr int CO = (CHG == 2) ? 3 : 0;     // first channel handled
    extern __shared__ float sm[];
    const int b = blockIdx.y;
    if (act && !act[b]) return;
    const int tid = threadIdx.x;
    const int tx0 = (blockIdx.x % tilesX) * TX, ty0 = (blockIdx.x / tilesX) * TY;
    const int WC = TX + 2 * rx, WR = TY + 2 * ry;
    const int WCP = WC + 1;
    float *inbuf = sm;                       // [NC][WR][WCP]
    float *xbuf = inbuf + NC * WR * WCP;     // [NC][WR][TX]
    __shared__ float stx[24], sty[24], stz[16];
    if (tid < 2 * rx + 1) stx[tid] = tapx[tid];
    if (tid < 2 * ry + 1) sty[tid] = tapy[tid];
    if (tid < 2 * RZ + 1) stz[tid] = tapz[tid];
    const float *w = warped + (size_t)b * N;
    const int cx = tid % TX, cy = tid / TX;
    const int xo = tx0 + cx, yo = ty0 + cy;
    const bool colIn = (xo < nx && yo < ny);

    // skip tiles without any valid voxel; otherwise find the z range [zmin, zmax] holding valid voxels (outputs are only used there)
    __shared__ int szmin, szmax;
    if (tid == 0) { szmin = nz; szmax = -1; }
    __syncthreads();
    {
        int lo = nz, hi = -1;
        if (colIn) {
            for (int z = 0; z < nz; z++) {
                size_t id = ((size_t)z * ny + yo) * nx + xo;
                if (valid_vox(ref, mask, w[id], id)) { lo = z; break; }
            }
            if (lo < nz)
                for (int z = nz - 1; z >= lo; z--) {
                    size_t id = ((size_t)z * ny + yo) * nx + xo;
                    if (valid_vox(ref, mask, w[id], id)) { hi = z; break; }
                }
        }
        if (lo < nz) { atomicMin(&szmin, lo); atomicMax(&szmax, hi); }
    }
    __syncthreads();
    const int zmin = szmin, zmax = szmax;
    if (zmax < 0) {
        if (partial && tid == 0) {
            partial[((size_t)b * nblk + blockIdx.x) * 2] = 0.0;
            partial[((size_t)b * nblk + blockIdx.x) * 2 + 1] = 0.0;
        }
        return;
    }

    float ring[S][NC];
#pragma unroll
    for (int s = 0; s < S; s++)
#pragma unroll
        for (int c = 0; c < NC; c++) ring[s][c] = 0.f;
    double dsum = 0.0, dcnt = 0.0;

    for (int zb = zmin - RZ; zb <= zmax + RZ; zb += S) {
#pragma unroll
        for (int j = 0; j < S; j++) {
            const int zi = zb + j;
            if (zi >= 0 && zi < nz && zi <= zmax + RZ) {
                // ---- plane window -> channels
                const size_t pl = (size_t)zi * nx * ny;
                for (int i = tid; i < WR * WC; i += NTHR) {
                    int row = i / WC, col = i - row * WC;
                    int x = tx0 - rx + col, y = ty0 - ry + row;
                    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, c4 = 0.f, c5 = 0.f;
                    if (x >= 0 && x < nx && y >= 0 && y < ny) {
                        size_t id = pl + (size_t)y * nx + x;
                        float wv = w[id];
                        float r = ref[id];
                        bool m = (mask == nullptr || mask[id] >= 0) && (r == r) && (wv == wv);
                        if (m) { c0 = 1.f; c1 = r; c2 = r * r; c3 = wv; c4 = wv * wv; c5 = r * wv; }
                    }
                    float *p = inbuf + row * WCP + col;
                    if (CHG == 0) { p[0] = c0; p[WR * WCP] = c1; p[2 * WR * WCP] = c2; p[3 * WR * WCP] = c3; p[4 * WR * WCP] = c4; p[5 * WR * WCP] = c5; }
                    else if (CHG == 1) { p[0] = c0; p[WR * WCP] = c1; p[2 * WR * WCP] = c2; }
                    else { p[0] = c3; p[WR * WCP] = c4; p[2 * WR * WCP] = c5; }
                }
                __syncthreads();
                // ---- x pass (three independent accumulation chains per thread for instruction level parallelism)
                {
                    const int total = NC * WR * TX;
                    for (int i0 = tid; i0 < total; i0 += 3 * NTHR) {
                        float hi[3] = {0.f, 0.f, 0.f}, lo[3] = {0.f, 0.f, 0.f};
                        const float *pp[3];
                        int ii[3];
#pragma unroll
                        for (int u = 0; u < 3; u++) {
                            int i = i0 + u * NTHR;
                            ii[u] = i;
                            int ic = i < total ? i : 0;
                            int c = ic / (WR * TX), rem = ic - c * (WR * TX);
                            int row = rem / TX, col = rem - row * TX;
                            pp[u] = inbuf + (c * WR + row) * WCP + col;
                        }
                        for (int k = 0; k <= 2 * rx; k++) {
                            float wk = stx[k];
#pragma unroll
                            for (int u = 0; u < 3; u++) acc_tap<EXACT>(hi[u], lo[u], wk, pp[u][k]);
                        }
#pragma unroll
                        for (int u = 0; u < 3; u++)
                            if (ii[u] < total) xbuf[ii[u]] = EXACT ? hi[u] + lo[u] : hi[u];
                    }
                }
                __syncthreads();
                // ---- y pass -> ring slot j (channels interleaved)
                {
                    float hi[NC], lo[NC];
#pragma unroll
                    for (int c = 0; c < NC; c++) { hi[c] = 0.f; lo[c] = 0.f; }
                    const float *p = xbuf + cy * TX + cx;
                    for (int k = 0; k <= 2 * ry; k++) {
                        float wk = sty[k];
#pragma unroll
                        for (int c = 0; c < NC; c++) acc_tap<EXACT>(hi[c], lo[c], wk, p[c * WR * TX + k * TX]);
                    }
#pragma unroll
                    for (int c = 0; c < NC; c++) ring[j][c] = EXACT ? hi[c] + lo[c] : hi[c];
                }
            } else {
#pragma unroll
                for (int c = 0; c < NC; c++) ring[j][c] = 0.f;
            }
            // ---- z pass for the output plane zo = zi - RZ
            const int zo = zi - RZ;
            if (zo >= zmin && zo <= zmax) {
                float o[NC];
                {
                    float hi[NC], lo[NC];
#pragma unroll
                    for (int c = 0; c < NC; c++) { hi[c] = 0.f; lo[c] = 0.f; }
#pragma unroll
                    for (int k = 0; k < S; k++) {
                        float wk = stz[k];
#pragma unroll
                        for (int c = 0; c < NC; c++) acc_tap<EXACT>(hi[c], lo[c], wk, ring[(j + 1 + k) % S][c]);
                    }
#pragma unroll
                    for (int c = 0; c < NC; c++) o[c] = EXACT ? hi[c] + lo[c] : hi[c];
                }
                if (colIn) {
                    size_t id = ((size_t)zo * ny + yo) * nx + xo;
                    float wv = w[id];
                    if (valid_vox(ref, mask, wv, id)) {
                        float *sb = stats + (size_t)b * 6 * N;
                        if (CHG == 0) {
                            if (STORE) {
#pragma unroll
                                for (int c = 0; c < 6; c++) sb[c * N + id] = o[c];
                            }
                            Stats s = stats_from(o);
                            float lv = (s.corr - (s.mr * s.mw)) / (s.sr * s.sw);
                            if (lv == lv && !isinf(lv)) { dsum += fabs((double)lv); dcnt += 1.0; }
                        } else if (CHG == 1) {
#pragma unroll
                            for (int c = 0; c < 3; c++) sb[c * N + id] = o[c];
                        } else {
                            float f6[6] = {sb[id], sb[N + id], sb[2 * N + id], o[0], o[1], o[2]};
                            if (STORE) {
#pragma unroll
                                for (int c = 0; c < 3; c++) sb[(3 + c) * N + id] = o[c];
                            }
                            Stats s = stats_from(f6);
                            float lv = (s.corr - (s.mr * s.mw)) / (s.sr * s.sw);
                            if (lv == lv && !isinf(lv)) { dsum += fabs((double)lv); dcnt += 1.0; }
                        }
                    }
                }
            }
        }
    }
    __shared__ double shd[32];
    dsum = block_sum_d(dsum, shd);
    dcnt = block_sum_d(dcnt, shd);
    if (tid == 0 && partial) {
        partial[((size_t)b * nblk + blockIdx.x) * 2] = dsum;
        partial[((size_t)b * nblk + blockIdx.x) * 2 + 1] = dcnt;
    }
}


// ======================================================================================================================
// Version 2 of the streaming kernel: compile-time radii (fully unrolled taps, tap weights in the constant bank through the
// kernel parameters), ring buffer shifted in registers (the plane loop is not unrolled, so the code stays small), static
// shared memory. Same arithmetic (same tap order, same exact accumulation) as the generic kernel above -> bit-identical.
#ifndef L2_TX
#define L2_TX 16
#endif
#ifndef L2_TY
#define L2_TY 16
#endif
#ifndef L2_SPLIT
#define L2_SPLIT 1
#endif
#ifndef L2_MINB
#define L2_MINB 3
#endif
// exact accumulation of a float product (Knuth two-sum; min/max based Fast2Sum was measured slower)
template <bool EX> __device__ __forceinline__ void acc_tap2(float &hi, float &lo, float w, float v) { acc_tap<EX>(hi, lo, w, v); }
template <bool EX> __device__ __forceinline__ float fin2(float hi, float lo) { return EX ? hi + lo : hi; }
// local NCC of one voxel from the six smoothed channels (not inlined: keeps the epilogue out of the register allocation of the filtering loops)
__device__ __noinline__ float lncc_local(float a0, float a1, float a2, float a3, float a4, float a5)
{
    float f6[6] = {a0, a1, a2, a3, a4, a5};
    Stats s = stats_from(f6);
    return (s.corr - (s.mr * s.mw)) / (s.sr * s.sw);
}
// gradient temporaries of one voxel (lncc_gprep_k arithmetic: double precision like the CPU code); ok = counted voxel
struct GpOut { float t1, t2, t3; int ok; };
__device__ __noinline__ GpOut lncc_gp_temps(float a0, float a1, float a2, float a3, float a4, float a5)
{
    float f6[6] = {a0, a1, a2, a3, a4, a5};
    Stats s = stats_from(f6);
    double refMean = s.mr, warMean = s.mw, refSdev = s.sr, warSdev = s.sw;
    double correla = (double)s.corr - (refMean * warMean);
    double temp1 = 1.0 / (refSdev * warSdev);
    double temp2 = correla / (refSdev * warSdev * warSdev * warSdev);
    double temp3 = (correla * warMean) / (refSdev * warSdev * warSdev * warSdev) - refMean / (refSdev * warSdev);
    GpOut o = {0.f, 0.f, 0.f, 0};
    if (temp1 == temp1 && !isinf(temp1) && temp2 == temp2 && !isinf(temp2) && temp3 == temp3 && !isinf(temp3)) {
        if (correla < 0) { temp1 *= -1.; temp2 *= -1.; temp3 *= -1.; }
        o.t1 = (float)temp1; o.t2 = (float)temp2; o.t3 = (float)temp3; o.ok = 1;
    }
    return o;
}
struct TapArgs { float x[17]; float y[17]; float z[13]; };

template <int RX, int RY, int RZ, int TW, int TH, int CHG, bool STORE, int MINB, bool EX>
__global__ void __launch_bounds__(TW *TH, MINB)
    lncc2_k(const float *__restrict__ ref, const int *__restrict__ mask, const float *__restrict__ warped, size_t N, int nx, int ny,
            int nz, const TapArgs ta, double *partial, int nblk, float *stats, const int *act, int tilesX, float *gp)
{
    constexpr int NT = TW * TH, S = 2 * RZ + 1, NC = (CHG == 0) ? 6 : 3;
    constexpr int WC = TW + 2 * RX, WR = TH + 2 * RY, WCP = WC + 1;
    __shared__ float inbuf[NC * WR * WCP];
    __shared__ float xbuf[NC * WR * TW];
    __shared__ unsigned char mbuf[WR * WC];   // validity flag of every window voxel (the centre of the window = the output voxel)
    const int b = blockIdx.y;
    if (act && !act[b]) return;
    const int tid = threadIdx.x;
    const int tx0 = (blockIdx.x % tilesX) * TW, ty0 = (blockIdx.x / tilesX) * TH;
    const float *w = warped + (size_t)b * N;
    const int cx = tid % TW, cy = tid / TW;
    const int xo = tx0 + cx, yo = ty0 + cy;
    const bool colIn = (xo < nx && yo < ny);

    __shared__ int szmin, szmax;
    if (tid == 0) { szmin = nz; szmax = -1; }
    __syncthreads();
    {
        int lo = nz, hi = -1;
        if (colIn) {
            for (int z = 0; z < nz; z++) {
                size_t id = ((size_t)z * ny + yo) * nx + xo;
                if (valid_vox(ref, mask, w[id], id)) { lo = z; break; }
            }
            if (lo < nz)
                for (int z = nz - 1; z >= lo; z--) {
                    size_t id = ((size_t)z * ny + yo) * nx + xo;
                    if (valid_vox(ref, mask, w[id], id)) { hi = z; break; }
                }
        }
        if (lo < nz) { atomicMin(&szmin, lo); atomicMax(&szmax, hi); }
    }
    __syncthreads();
    const int zmin = szmin, zmax = szmax;
    if (CHG == 2 && STORE && colIn) {   // gradient temporaries of voxels outside the processed z range (all invalid): zero
        float *gt = gp + (size_t)b * 6 * N;
        for (int z = 0; z < nz; z++) {
            if (z >= zmin && z <= zmax) continue;
            size_t id = ((size_t)z * ny + yo) * nx + xo;
            gt[N + id] = 0.f; gt[2 * N + id] = 0.f; gt[3 * N + id] = 0.f;
        }
    }
    if (zmax < 0) {
        if (partial && tid == 0) {
            partial[((size_t)b * nblk + blockIdx.x) * 2] = 0.0;
            partial[((size_t)b * nblk + blockIdx.x) * 2 + 1] = 0.0;
        }
        return;
    }

    float ring[S][NC];
#pragma unroll
    for (int s = 0; s < S; s++)
#pragma unroll
        for (int c = 0; c < NC; c++) ring[s][c] = 0.f;
    double dsum = 0.0, dcnt = 0.0;
    unsigned vmask = 0;   // bit k: the voxel of the k-th most recent plane (this column) is valid

    // software prefetch of the raw window values of the next plane (hides the global load latency behind the filtering)
    constexpr int NE = (WR * WC + NT - 1) / NT;
    float pw[NE], pr[NE];
    auto fetch = [&](int zq) {
        const size_t pl = (size_t)zq * nx * ny;
#pragma unroll
        for (int e = 0; e < NE; e++) {
            int i = tid + e * NT;
            pw[e] = nanf(""); pr[e] = 0.f;
            if (zq >= 0 && zq < nz && i < WR * WC) {
                int row = i / WC, col = i - row * WC;
                int x = tx0 - RX + col, y = ty0 - RY + row;
                if (x >= 0 && x < nx && y >= 0 && y < ny) {
                    size_t id = pl + (size_t)y * nx + x;
                    pw[e] = w[id]; pr[e] = ref[id];
                }
            }
        }
    };
    fetch(zmin - RZ);

    for (int zi = zmin - RZ; zi <= zmax + RZ; ++zi) {
        float nw[NC];
        const int zo = zi - RZ;
        const bool doOut = colIn && zo >= zmin && zo <= zmax;
        float es0 = 0.f, es1 = 0.f, es2 = 0.f;
        const size_t eid = ((size_t)zo * ny + yo) * nx + xo;
        float *sb = stats + (size_t)b * 6 * N;
        // (statistics of the first kernel are loaded where they are used)
        if (zi >= 0 && zi < nz) {
#pragma unroll
            for (int e = 0; e < NE; e++) {
                int i = tid + e * NT;
                if (i < WR * WC) {
                    int row = i / WC, col = i - row * WC;
                    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, c4 = 0.f, c5 = 0.f;
                    float wv = pw[e], r = pr[e];
                    bool m = (r == r) && (wv == wv);   // (kernels with a non trivial mask use the generic path)
                    if (m) { c0 = 1.f; c1 = r; c2 = r * r; c3 = wv; c4 = wv * wv; c5 = r * wv; }
                    mbuf[i] = m ? 1 : 0;
                    float *p = inbuf + row * WCP + col;
                    if (CHG == 0) { p[0] = c0; p[WR * WCP] = c1; p[2 * WR * WCP] = c2; p[3 * WR * WCP] = c3; p[4 * WR * WCP] = c4; p[5 * WR * WCP] = c5; }
                    else if (CHG == 1) { p[0] = c0; p[WR * WCP] = c1; p[2 * WR * WCP] = c2; }
                    else { p[0] = c3; p[WR * WCP] = c4; p[2 * WR * WCP] = c5; }
                }
            }
            fetch(zi + 1);
            __syncthreads();
            vmask = (vmask << 1) | (unsigned)mbuf[(RY + cy) * WC + RX + cx];
            {
                constexpr int total = NC * WR * TW;
#pragma unroll 1
                for (int i0 = tid; i0 < total; i0 += 3 * NT) {
                    float hi[3] = {0.f, 0.f, 0.f}, lo[3] = {0.f, 0.f, 0.f};
                    const float *pp[3];
#pragma unroll
                    for (int u = 0; u < 3; u++) {
                        int i = i0 + u * NT;
                        int ic = i < total ? i : 0;
                        int c = ic / (WR * TW), rem = ic - c * (WR * TW);
                        int row = rem / TW, col = rem - row * TW;
                        pp[u] = inbuf + (c * WR + row) * WCP + col;
                    }
#pragma unroll
                    for (int k = 0; k <= 2 * RX; k++) {
                        const float wk = ta.x[k];
#pragma unroll
                        for (int u = 0; u < 3; u++) acc_tap2<EX>(hi[u], lo[u], wk, pp[u][k]);
                    }
#pragma unroll
                    for (int u = 0; u < 3; u++) {
                        int i = i0 + u * NT;
                        if (i < total) xbuf[i] = fin2<EX>(hi[u], lo[u]);
                    }
                }
            }
            __syncthreads();
            {
                float hi[NC], lo[NC];
#pragma unroll
                for (int c = 0; c < NC; c++) { hi[c] = 0.f; lo[c] = 0.f; }
                const float *p = xbuf + cy * TW + cx;
#pragma unroll
                for (int k = 0; k <= 2 * RY; k++) {
                    const float wk = ta.y[k];
#pragma unroll
                    for (int c = 0; c < NC; c++) acc_tap2<EX>(hi[c], lo[c], wk, p[c * WR * TW + k * TW]);
                }
#pragma unroll
                for (int c = 0; c < NC; c++) nw[c] = fin2<EX>(hi[c], lo[c]);
            }
        } else {
            fetch(zi + 1);
            vmask <<= 1;
#pragma unroll
            for (int c = 0; c < NC; c++) nw[c] = 0.f;
        }
#pragma unroll
        for (int s = 0; s < S - 1; s++)
#pragma unroll
            for (int c = 0; c < NC; c++) ring[s][c] = ring[s + 1][c];
#pragma unroll
        for (int c = 0; c < NC; c++) ring[S - 1][c] = nw[c];
        if (zo >= zmin && zo <= zmax) {
            float o[NC];
            {
                float hi[NC], lo[NC];
#pragma unroll
                for (int c = 0; c < NC; c++) { hi[c] = 0.f; lo[c] = 0.f; }
#pragma unroll
                for (int k = 0; k < S; k++) {
                    const float wk = ta.z[k];
#pragma unroll
                    for (int c = 0; c < NC; c++) acc_tap2<EX>(hi[c], lo[c], wk, ring[k][c]);
                }
#pragma unroll
                for (int c = 0; c < NC; c++) o[c] = fin2<EX>(hi[c], lo[c]);
            }
            if (doOut) {
                const size_t id = eid;
                if ((vmask >> RZ) & 1u) {
                    if (CHG == 0) {
                        if (STORE) {
#pragma unroll
                            for (int c = 0; c < 6; c++) sb[c * N + id] = o[c];
                        }
                        Stats s = stats_from(o);
                        float lv = (s.corr - (s.mr * s.mw)) / (s.sr * s.sw);
                        if (lv == lv && !isinf(lv)) { dsum += fabs((double)lv); dcnt += 1.0; }
                    } else if (CHG == 1) {
#pragma unroll
                        for (int c = 0; c < 3; c++) sb[c * N + id] = o[c];
                    } else {
                        if (STORE) {
                            GpOut g = lncc_gp_temps(sb[eid], sb[N + eid], sb[2 * N + eid], o[0], o[1], o[2]);
                            if (g.ok) dsum += 1.0;
                            float *gt = gp + (size_t)b * 6 * N;
                            gt[N + eid] = g.t1; gt[2 * N + eid] = g.t2; gt[3 * N + eid] = g.t3;
                        } else {
                            float lv = lncc_local(sb[eid], sb[N + eid], sb[2 * N + eid], o[0], o[1], o[2]);
                            if (lv == lv && !isinf(lv)) { dsum += fabs((double)lv); dcnt += 1.0; }
                        }
                    }
                } else if (CHG == 2 && STORE) {
                    float *gt = gp + (size_t)b * 6 * N;
                    gt[N + id] = 0.f; gt[2 * N + id] = 0.f; gt[3 * N + id] = 0.f;
                }
            }
        }
    }
    __shared__ double shd[32];
    dsum = block_sum_d(dsum, shd);
    dcnt = block_sum_d(dcnt, shd);
    if (tid == 0 && partial) {
        partial[((size_t)b * nblk + blockIdx.x) * 2] = dsum;
        partial[((size_t)b * nblk + blockIdx.x) * 2 + 1] = dcnt;
    }
}


// ======================================================================================================================
// CHG1 (reference channels m, r, r^2) with reuse of template independent results: where the Gaussian window of a pass only holds
// voxels for which the warped image is valid, the smoothed reference channels equal the precomputed ones (Px, Pxy, Pxyz = x, x+y,
// x+y+z smoothing of the reference channels with every voxel valid, computed once per level with the same exact arithmetic).
// Same values bit for bit; the three passes skip their sums wherever the corresponding window is fully valid (x radius 8 only).
template <int RY, int RZ, bool STORE>
__global__ void __launch_bounds__(256, 2)
    lncc3_k(const float *__restrict__ ref, const int *__restrict__ mask, const float *__restrict__ warped, size_t N, int nx, int ny,
            int nz, const TapArgs ta, double *partial, int nblk, float *stats, const int *act, int tilesX,
            const float *__restrict__ Px, const float *__restrict__ Pxy, const float *__restrict__ Pxyz)
{
    constexpr int TW = 16, TH = 16, NT = 256, RX = 8, S = 2 * RZ + 1, NC = 3;
    constexpr int WC = TW + 2 * RX, WR = TH + 2 * RY, WCP = WC + 1;
    static_assert(WC == 32, "row masks need a 32 wide window");
    __shared__ float inbuf[NC * WR * WCP];
    __shared__ float xbuf[NC * WR * TW];
    __shared__ unsigned xmask[2][WR];
    const int b = blockIdx.x;   // templates fastest: the blocks working on the same tile (same reference tables) run together (L2 reuse)
    const int tile = blockIdx.y;
    if (act && !act[b]) return;
    const int tid = threadIdx.x;
    const int tx0 = (tile % tilesX) * TW, ty0 = (tile / tilesX) * TH;
    const float *w = warped + (size_t)b * N;
    const int cx = tid % TW, cy = tid / TW;
    const int xo = tx0 + cx, yo = ty0 + cy;
    const bool colIn = (xo < nx && yo < ny);

    __shared__ int szmin, szmax;
    if (tid == 0) { szmin = nz; szmax = -1; }
    __syncthreads();
    {
        int lo = nz, hi = -1;
        if (colIn) {
            for (int z = 0; z < nz; z++) {
                size_t id = ((size_t)z * ny + yo) * nx + xo;
                if (valid_vox(ref, mask, w[id], id)) { lo = z; break; }
            }
            if (lo < nz)
                for (int z = nz - 1; z >= lo; z--) {
                    size_t id = ((size_t)z * ny + yo) * nx + xo;
                    if (valid_vox(ref, mask, w[id], id)) { hi = z; break; }
                }
        }
        if (lo < nz) { atomicMin(&szmin, lo); atomicMax(&szmax, hi); }
    }
    __syncthreads();
    const int zmin = szmin, zmax = szmax;
    if (zmax < 0) {
        if (partial && tid == 0) {
            partial[((size_t)b * nblk + tile) * 2] = 0.0;
            partial[((size_t)b * nblk + tile) * 2 + 1] = 0.0;
        }
        return;
    }

    float ring[S][NC];
#pragma unroll
    for (int s = 0; s < S; s++)
#pragma unroll
        for (int c = 0; c < NC; c++) ring[s][c] = 0.f;
    unsigned fmask = 0xffffffffu;   // bit k: y-pass output of the k-th most recent plane equals Pxy (planes outside the image count as such)
    constexpr unsigned ALLZ = (S >= 32) ? 0xffffffffu : ((1u << S) - 1u);
    double dsum = 0.0, dcnt = 0.0;

    constexpr int NE = (WR * WC + NT - 1) / NT;
    static_assert((WR * WC) % 32 == 0, "window rows must be whole warps");
    float pw[NE], pr[NE];
    int pm[NE];
    auto fetch = [&](int zq) {
        const size_t pl = (size_t)zq * nx * ny;
#pragma unroll
        for (int e = 0; e < NE; e++) {
            int i = tid + e * NT;
            pw[e] = nanf(""); pr[e] = nanf(""); pm[e] = 0;
            if (zq >= 0 && zq < nz && i < WR * WC) {
                int row = i / WC, col = i - row * WC;
                int x = tx0 - RX + col, y = ty0 - RY + row;
                if (x >= 0 && x < nx && y >= 0 && y < ny) {
                    size_t id = pl + (size_t)y * nx + x;
                    pw[e] = w[id]; pr[e] = ref[id]; pm[e] = mask ? mask[id] : 0;
                }
            }
        }
    };
    fetch(zmin - RZ);

    for (int zi = zmin - RZ; zi <= zmax + RZ; ++zi) {
        float nw[NC];
        const int zo = zi - RZ;
        const bool doOut = colIn && zo >= zmin && zo <= zmax;
        float ew = 0.f, er = 0.f;
        int em = 0;
        const size_t eid = ((size_t)zo * ny + yo) * nx + xo;
        float *sb = stats + (size_t)b * 6 * N;
        if (doOut) { ew = w[eid]; er = ref[eid]; em = mask ? mask[eid] : 0; }
        float qx[2][3], qy[3], qz[3];
        const size_t pl = (size_t)zi * nx * ny;
        if (zi >= 0 && zi < nz) {
#pragma unroll
            for (int it = 0; it < 2; it++) {
                int idx = tid + it * NT;
                int row = idx / TW, col = idx - row * TW;
                int yrow = ty0 - RY + row, xcol = tx0 + col;
                bool ok = idx < WR * TW && yrow >= 0 && yrow < ny && xcol < nx;
                size_t g = pl + (size_t)yrow * nx + xcol;
#pragma unroll
                for (int c = 0; c < 3; c++) qx[it][c] = ok ? Px[c * N + g] : 0.f;
            }
            {
                size_t g = pl + (size_t)yo * nx + xo;
#pragma unroll
                for (int c = 0; c < 3; c++) qy[c] = colIn ? Pxy[c * N + g] : 0.f;
            }
        }
        if (doOut) {
#pragma unroll
            for (int c = 0; c < 3; c++) qz[c] = Pxyz[c * N + eid];
        }
        if (zi >= 0 && zi < nz) {
#pragma unroll
            for (int e = 0; e < NE; e++) {
                int i = tid + e * NT;
                if (i < WR * WC) {
                    int row = i / WC, col = i - row * WC;
                    float c0 = 0.f, c1 = 0.f, c2 = 0.f;
                    float wv = pw[e], r = pr[e];
                    bool mref = (mask == nullptr || pm[e] >= 0) && (r == r);
                    bool wok = (wv == wv);
                    if (mref && wok) { c0 = 1.f; c1 = r; c2 = r * r; }
                    float *p = inbuf + row * WCP + col;
                    p[0] = c0; p[WR * WCP] = c1; p[2 * WR * WCP] = c2;
                    unsigned bm = __ballot_sync(0xffffffffu, mref && !wok);   // bit col: valid reference voxel whose warped value is NaN
                    if (col == 0) {
                        unsigned a = bm | (bm >> 1);
                        a |= a >> 2; a |= a >> 4; a |= a >> 8; a |= bm >> 16;
                        xmask[zi & 1][row] = ~a & 0xffffu;                  // bit c: x window of output column c has no such voxel
                    }
                }
            }
            fetch(zi + 1);
            __syncthreads();
            // ---- x pass: (row, col) per thread, the three channels as independent chains
#pragma unroll
            for (int it = 0; it < 2; it++) {
                int idx = tid + it * NT;
                if (idx < WR * TW) {
                    int row = idx / TW, col = idx - row * TW;
                    int yrow = ty0 - RY + row, xcol = tx0 + col;
                    if (yrow < 0 || yrow >= ny) {
                        xbuf[idx] = 0.f; xbuf[WR * TW + idx] = 0.f; xbuf[2 * WR * TW + idx] = 0.f;
                    } else if (((xmask[zi & 1][row] >> col) & 1u) && xcol < nx) {
                        xbuf[idx] = qx[it][0]; xbuf[WR * TW + idx] = qx[it][1]; xbuf[2 * WR * TW + idx] = qx[it][2];
                    } else {
                        float hi[3] = {0.f, 0.f, 0.f}, lo[3] = {0.f, 0.f, 0.f};
                        const float *pp = inbuf + row * WCP + col;
#pragma unroll
                        for (int k = 0; k <= 2 * RX; k++) {
                            const float wk = ta.x[k];
#pragma unroll
                            for (int c = 0; c < 3; c++) acc_tap2<true>(hi[c], lo[c], wk, pp[c * WR * WCP + k]);
                        }
#pragma unroll
                        for (int c = 0; c < 3; c++) xbuf[c * WR * TW + idx] = fin2<true>(hi[c], lo[c]);
                    }
                }
            }
            __syncthreads();
            // ---- y pass
            {
                unsigned all = 0xffffffffu;
#pragma unroll
                for (int k = 0; k <= 2 * RY; k++) all &= xmask[zi & 1][cy + k];
                const bool yfull = colIn && ((all >> cx) & 1u);
                fmask = (fmask << 1) | (yfull ? 1u : 0u);
                if (yfull) {
                    nw[0] = qy[0]; nw[1] = qy[1]; nw[2] = qy[2];
                } else {
                    float hi[NC], lo[NC];
#pragma unroll
                    for (int c = 0; c < NC; c++) { hi[c] = 0.f; lo[c] = 0.f; }
                    const float *p = xbuf + cy * TW + cx;
#pragma unroll
                    for (int k = 0; k <= 2 * RY; k++) {
                        const float wk = ta.y[k];
#pragma unroll
                        for (int c = 0; c < NC; c++) acc_tap2<true>(hi[c], lo[c], wk, p[c * WR * TW + k * TW]);
                    }
#pragma unroll
                    for (int c = 0; c < NC; c++) nw[c] = fin2<true>(hi[c], lo[c]);
                }
            }
        } else {
            fetch(zi + 1);
            fmask = (fmask << 1) | 1u;
#pragma unroll
            for (int c = 0; c < NC; c++) nw[c] = 0.f;
        }
#pragma unroll
        for (int s = 0; s < S - 1; s++)
#pragma unroll
            for (int c = 0; c < NC; c++) ring[s][c] = ring[s + 1][c];
#pragma unroll
        for (int c = 0; c < NC; c++) ring[S - 1][c] = nw[c];
        if (zo >= zmin && zo <= zmax) {
            float o[NC];
            if (doOut && ((fmask & ALLZ) == ALLZ)) {
                o[0] = qz[0]; o[1] = qz[1]; o[2] = qz[2];
            } else {
                float hi[NC], lo[NC];
#pragma unroll
                for (int c = 0; c < NC; c++) { hi[c] = 0.f; lo[c] = 0.f; }
#pragma unroll
                for (int k = 0; k < S; k++) {
                    const float wk = ta.z[k];
#pragma unroll
                    for (int c = 0; c < NC; c++) acc_tap2<true>(hi[c], lo[c], wk, ring[k][c]);
                }
#pragma unroll
                for (int c = 0; c < NC; c++) o[c] = fin2<true>(hi[c], lo[c]);
            }
            if (doOut) {
                if ((mask == nullptr || em >= 0) && (er == er) && (ew == ew)) {
#pragma unroll
                    for (int c = 0; c < 3; c++) sb[c * N + eid] = o[c];
                }
            }
        }
    }
    __shared__ double shd[32];
    dsum = block_sum_d(dsum, shd);
    dcnt = block_sum_d(dcnt, shd);
    if (tid == 0 && partial) {
        partial[((size_t)b * nblk + tile) * 2] = dsum;
        partial[((size_t)b * nblk + tile) * 2 + 1] = dcnt;
    }
}

template <int RX, int RY, int RZ, bool STORE, int CHG, bool EX>
static void launch2_one(LnccCtx &c, const float *warped, float *stats, const int *act, cudaStream_t st)
{
    TapArgs ta;
    memset(&ta, 0, sizeof ta);
    for (int k = 0; k <= 2 * RX; k++) ta.x[k] = c.gt[0].h[k];
    for (int k = 0; k <= 2 * RY; k++) ta.y[k] = c.gt[1].h[k];
    for (int k = 0; k <= 2 * RZ; k++) ta.z[k] = c.gt[2].h[k];
    int tilesX = cdiv(c.dims[0], L2_TX), tilesY = cdiv(c.dims[1], L2_TY);
    dim3 grid(tilesX * tilesY, c.B);
    if (CHG == 1 && RX == 8 && L2_TX == 16 && L2_TY == 16 && c.Pxyz) {
        static const bool off = getenv("F3D_NOSKIP") != nullptr;
        if (!off) {
            lncc3_k<RY, RZ, STORE><<<dim3(c.B, tilesX * tilesY), 256, 0, st>>>(c.ref, c.mask, warped, c.N, c.dims[0], c.dims[1], c.dims[2], ta, c.partial, tilesX * tilesY, stats,
                                                         act, tilesX, c.Px, c.Pxy, c.Pxyz);
            c.nblk_used = tilesX * tilesY;
            return;
        }
    }
    // STORE with the split kernels: the second kernel produces the gradient temporaries directly (into bufA channels 1..3) instead of
    // storing its statistics; the count of finite voxels goes to the partial sums like lncc_gprep_k did
    lncc2_k<RX, RY, RZ, L2_TX, L2_TY, CHG, STORE, L2_MINB, EX><<<grid, L2_TX * L2_TY, 0, st>>>(
        c.ref, c.mask, warped, c.N, c.dims[0], c.dims[1], c.dims[2], ta, c.partial, tilesX * tilesY, stats, act, tilesX, c.bufA);
    if (CHG == 2 && STORE) c.gp_done = true;
    c.nblk_used = tilesX * tilesY;
}
template <int RX, int RY, int RZ, bool STORE, bool EX>
static void launch2(LnccCtx &c, const float *warped, float *stats, const int *act, cudaStream_t st)
{
    if constexpr (L2_SPLIT) {
        launch2_one<RX, RY, RZ, STORE, 1, EX>(c, warped, stats, act, st);
        launch2_one<RX, RY, RZ, STORE, 2, EX>(c, warped, stats, act, st);
    } else launch2_one<RX, RY, RZ, STORE, 0, EX>(c, warped, stats, act, st);
}
template <bool STORE, bool EX>
static bool dispatch2(LnccCtx &c, const float *warped, float *stats, const int *act, cudaStream_t st)
{
    static const bool off = getenv("F3D_LNCC_V1") != nullptr;
    if (off || c.mask) return false;   // the compile-time radius kernels assume that every voxel is active (mask == nullptr)
    int rx = c.gt[0].radius, ry = c.gt[1].radius, rz = c.gt[2].radius;
#define D2(X, Y, Z) if (rx == X && ry == Y && rz == Z) { launch2<X, Y, Z, STORE, EX>(c, warped, stats, act, st); return true; }
    D2(8, 8, 6) D2(8, 7, 6) D2(4, 4, 3) D2(4, 3, 3) D2(2, 2, 3) D2(2, 1, 3)
#undef D2
    return false;
}


// ======================================================================================================================
// Gradient phase: Gaussian smoothing of the three LNCC gradient temporaries (x, y, z passes through shared memory and a register ring,
// same exact arithmetic as the generic convolutions) fused with NiftyReg's per voxel combination (what lncc_gfinal_k computes):
//   voxgrad = -(wgrad * (t1*ref - t2*warped + t3) / count)   at valid voxels, 0 elsewhere.
// temps [3][N] per template at stride 6N (bufA + N), dens = smoothed density (channel 0 of the statistics buffer), acc = per template
// {.., .., count, ..}.
template <int RX, int RY, int RZ, bool EX>
__global__ void __launch_bounds__(256, 2)
    lncc_smooth_k(const float *__restrict__ ref, const int *__restrict__ mask, const float *__restrict__ warped, size_t N, int nx, int ny,
                  int nz, const TapArgs ta, const float *__restrict__ temps, const float *__restrict__ dens,
                  const float *__restrict__ wgrad, float *__restrict__ out, const double *__restrict__ acc, const int *act, int tilesX)
{
    constexpr int TW = 16, TH = 16, NT = 256, S = 2 * RZ + 1, NC = 3;
    constexpr int WC = TW + 2 * RX, WR = TH + 2 * RY, WCP = WC + 1;
    __shared__ float inbuf[NC * WR * WCP];
    __shared__ float xbuf[NC * WR * TW];
    const int b = blockIdx.y;
    if (act && !act[b]) return;
    const int tid = threadIdx.x;
    const int tx0 = (blockIdx.x % tilesX) * TW, ty0 = (blockIdx.x / tilesX) * TH;
    const float *w = warped + (size_t)b * N;
    const float *tp = temps + (size_t)b * 6 * N;
    const float *dn = dens + (size_t)b * 6 * N;
    const float *wg = wgrad + (size_t)b * 3 * N;
    float *og = out + (size_t)b * 3 * N;
    const int cx = tid % TW, cy = tid / TW;
    const int xo = tx0 + cx, yo = ty0 + cy;
    const bool colIn = (xo < nx && yo < ny);

    __shared__ int szmin, szmax;
    if (tid == 0) { szmin = nz; szmax = -1; }
    __syncthreads();
    {
        int lo = nz, hi = -1;
        if (colIn) {
            for (int z = 0; z < nz; z++) {
                size_t id = ((size_t)z * ny + yo) * nx + xo;
                if (valid_vox(ref, mask, w[id], id)) { lo = z; break; }
            }
            if (lo < nz)
                for (int z = nz - 1; z >= lo; z--) {
                    size_t id = ((size_t)z * ny + yo) * nx + xo;
                    if (valid_vox(ref, mask, w[id], id)) { hi = z; break; }
                }
        }
        if (lo < nz) { atomicMin(&szmin, lo); atomicMax(&szmax, hi); }
    }
    __syncthreads();
    const int zmin = szmin, zmax = szmax;
    if (colIn) {   // outputs outside [zmin, zmax] belong to invalid voxels: zero gradient
        for (int z = 0; z < nz; z++) {
            if (z >= zmin && z <= zmax) continue;
            size_t id = ((size_t)z * ny + yo) * nx + xo;
            og[id] = 0.f; og[N + id] = 0.f; og[2 * N + id] = 0.f;
        }
    }
    if (zmax < 0) return;
    const double adjusted = 1.0 / acc[(size_t)b * 4 + 2];

    float ring[S][NC];
#pragma unroll
    for (int s = 0; s < S; s++)
#pragma unroll
        for (int c = 0; c < NC; c++) ring[s][c] = 0.f;

    constexpr int NE = (WR * WC + NT - 1) / NT;
    float p0[NE], p1[NE], p2[NE];
    auto fetch = [&](int zq) {
        const size_t pl = (size_t)zq * nx * ny;
#pragma unroll
        for (int e = 0; e < NE; e++) {
            int i = tid + e * NT;
            p0[e] = 0.f; p1[e] = 0.f; p2[e] = 0.f;
            if (zq >= 0 && zq < nz && i < WR * WC) {
                int row = i / WC, col = i - row * WC;
                int x = tx0 - RX + col, y = ty0 - RY + row;
                if (x >= 0 && x < nx && y >= 0 && y < ny) {
                    size_t id = pl + (size_t)y * nx + x;
                    p0[e] = tp[N + id]; p1[e] = tp[2 * N + id]; p2[e] = tp[3 * N + id];
                }
            }
        }
    };
    fetch(zmin - RZ);

    for (int zi = zmin - RZ; zi <= zmax + RZ; ++zi) {
        float nw[NC];
        const int zo = zi - RZ;
        const bool doOut = colIn && zo >= zmin && zo <= zmax;
        const size_t eid = ((size_t)zo * ny + yo) * nx + xo;
        float ew = 0.f, er = 0.f, ed = 0.f, eg0 = 0.f, eg1 = 0.f, eg2 = 0.f;
        int em = 0;
        if (doOut) {
            ew = w[eid]; er = ref[eid]; em = mask ? mask[eid] : 0;
            ed = dn[eid]; eg0 = wg[eid]; eg1 = wg[N + eid]; eg2 = wg[2 * N + eid];
        }
        if (zi >= 0 && zi < nz) {
#pragma unroll
            for (int e = 0; e < NE; e++) {
                int i = tid + e * NT;
                if (i < WR * WC) {
                    int row = i / WC, col = i - row * WC;
                    float *p = inbuf + row * WCP + col;
                    p[0] = p0[e]; p[WR * WCP] = p1[e]; p[2 * WR * WCP] = p2[e];
                }
            }
            fetch(zi + 1);
            __syncthreads();
            {
                constexpr int total = NC * WR * TW;
#pragma unroll 1
                for (int i0 = tid; i0 < total; i0 += 3 * NT) {
                    float hi[3] = {0.f, 0.f, 0.f}, lo[3] = {0.f, 0.f, 0.f};
                    const float *pp[3];
#pragma unroll
                    for (int u = 0; u < 3; u++) {
                        int i = i0 + u * NT;
                        int ic = i < total ? i : 0;
                        int c = ic / (WR * TW), rem = ic - c * (WR * TW);
                        int row = rem / TW, col = rem - row * TW;
                        pp[u] = inbuf + (c * WR + row) * WCP + col;
                    }
#pragma unroll
                    for (int k = 0; k <= 2 * RX; k++) {
                        const float wk = ta.x[k];
#pragma unroll
                        for (int u = 0; u < 3; u++) acc_tap2<EX>(hi[u], lo[u], wk, pp[u][k]);
                    }
#pragma unroll
                    for (int u = 0; u < 3; u++) {
                        int i = i0 + u * NT;
                        if (i < total) xbuf[i] = fin2<EX>(hi[u], lo[u]);
                    }
                }
            }
            __syncthreads();
            {
                float hi[NC], lo[NC];
#pragma unroll
                for (int c = 0; c < NC; c++) { hi[c] = 0.f; lo[c] = 0.f; }
                const float *p = xbuf + cy * TW + cx;
#pragma unroll
                for (int k = 0; k <= 2 * RY; k++) {
                    const float wk = ta.y[k];
#pragma unroll
                    for (int c = 0; c < NC; c++) acc_tap2<EX>(hi[c], lo[c], wk, p[c * WR * TW + k * TW]);
                }
#pragma unroll
                for (int c = 0; c < NC; c++) nw[c] = fin2<EX>(hi[c], lo[c]);
            }
        } else {
            fetch(zi + 1);
#pragma unroll
            for (int c = 0; c < NC; c++) nw[c] = 0.f;
        }
#pragma unroll
        for (int s = 0; s < S - 1; s++)
#pragma unroll
            for (int c = 0; c < NC; c++) ring[s][c] = ring[s + 1][c];
#pragma unroll
        for (int c = 0; c < NC; c++) ring[S - 1][c] = nw[c];
        if (zo >= zmin && zo <= zmax) {
            float o[NC];
            {
                float hi[NC], lo[NC];
#pragma unroll
                for (int c = 0; c < NC; c++) { hi[c] = 0.f; lo[c] = 0.f; }
#pragma unroll
                for (int k = 0; k < S; k++) {
                    const float wk = ta.z[k];
#pragma unroll
                    for (int c = 0; c < NC; c++) acc_tap2<EX>(hi[c], lo[c], wk, ring[k][c]);
                }
#pragma unroll
                for (int c = 0; c < NC; c++) o[c] = fin2<EX>(hi[c], lo[c]);
            }
            if (doOut) {
                float g0 = 0.f, g1 = 0.f, g2 = 0.f;
                if ((mask == nullptr || em >= 0) && (er == er) && (ew == ew)) {
                    float t1 = o[0] / ed, t2 = o[1] / ed, t3 = o[2] / ed;
                    float cm = t1 * er - t2 * ew + t3;
                    double common = (double)cm;
                    common *= adjusted;
                    g0 = (float)(0.0 - (double)eg0 * common);
                    g1 = (float)(0.0 - (double)eg1 * common);
                    g2 = (float)(0.0 - (double)eg2 * common);
                    if (g0 != g0 || isinf(g0)) g0 = 0.f;
                    if (g1 != g1 || isinf(g1)) g1 = 0.f;
                    if (g2 != g2 || isinf(g2)) g2 = 0.f;
                }
                og[eid] = g0; og[N + eid] = g1; og[2 * N + eid] = g2;
            }
        }
    }
}

template <int RX, int RY, int RZ, bool EX>
static void launch_smooth(LnccCtx &c, const float *warped, const float *wgrad, float *voxgrad, const int *act, cudaStream_t st)
{
    TapArgs ta;
    memset(&ta, 0, sizeof ta);
    for (int k = 0; k <= 2 * RX; k++) ta.x[k] = c.gt[0].h[k];
    for (int k = 0; k <= 2 * RY; k++) ta.y[k] = c.gt[1].h[k];
    for (int k = 0; k <= 2 * RZ; k++) ta.z[k] = c.gt[2].h[k];
    int tilesX = cdiv(c.dims[0], 16), tilesY = cdiv(c.dims[1], 16);
    lncc_smooth_k<RX, RY, RZ, EX><<<dim3(tilesX * tilesY, c.B), 256, 0, st>>>(c.ref, c.mask, warped, c.N, c.dims[0], c.dims[1], c.dims[2], ta, c.bufA,
                                                                           c.bufB, wgrad, voxgrad, c.acc, act, tilesX);
}
bool lncc_gradient_smooth_fused(LnccCtx &c, const float *warped, const float *wgrad, float *voxgrad, const int *act, cudaStream_t st)
{
    static const bool off = getenv("F3D_NOFUSE_GRAD") != nullptr;
    if (off || c.mask) return false;
    if (!c.exact && lnccf_smooth(c, warped, wgrad, voxgrad, act, st)) return true;
    int rx = c.gt[0].radius, ry = c.gt[1].radius, rz = c.gt[2].radius;
#define D3(X, Y, Z) if (rx == X && ry == Y && rz == Z) { if (c.exact) launch_smooth<X, Y, Z, true>(c, warped, wgrad, voxgrad, act, st); else launch_smooth<X, Y, Z, false>(c, warped, wgrad, voxgrad, act, st); return true; }
    D3(8, 8, 6) D3(8, 7, 6) D3(4, 4, 3) D3(4, 3, 3) D3(2, 2, 3) D3(2, 1, 3)
#undef D3
    return false;
}

template <int S, bool EXACT, bool STORE, int CHG>
static void launch_one(LnccCtx &c, const float *warped, float *stats, const int *act, cudaStream_t st)
{
    int rx = c.gt[0].radius, ry = c.gt[1].radius;
    int tilesX = cdiv(c.dims[0], TX), tilesY = cdiv(c.dims[1], TY);
    int nc = (CHG == 0) ? 6 : 3;
    size_t smem = ((size_t)nc * (TY + 2 * ry) * (TX + 2 * rx + 1) + (size_t)nc * (TY + 2 * ry) * TX) * sizeof(float);
    auto kern = lncc_stream_k<S, EXACT, STORE, CHG>;
    static bool attr = false;
    if (!attr) { CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); attr = true; }
    dim3 grid(tilesX * tilesY, c.B);
    kern<<<grid, NTHR, smem, st>>>(c.ref, c.mask, warped, c.N, c.dims[0], c.dims[1], c.dims[2], rx, ry, c.gt[0].d, c.gt[1].d, c.gt[2].d,
                                   c.partial, c.nblk_s, stats, act, tilesX);
}
// scratch: [B][6][N] buffer used by the two-kernel (3+3 channels) variant that keeps the register ring small
template <int S, bool EXACT, bool STORE>
static void launch_stream(LnccCtx &c, const float *warped, float *stats, const int *act, cudaStream_t st)
{
    launch_one<S, EXACT, STORE, 1>(c, warped, stats, act, st);
    launch_one<S, EXACT, STORE, 2>(c, warped, stats, act, st);
}

template <bool EXACT, bool STORE>
static bool dispatch(LnccCtx &c, const float *warped, float *stats, const int *act, cudaStream_t st)
{
    if (dispatch2<STORE, EXACT>(c, warped, stats, act, st)) return true;
    c.nblk_used = c.nblk_s;
    int rz = c.gt[2].radius;
    if (c.gt[0].radius > 11 || c.gt[1].radius > 11) return false;
    if (rz == 3) { launch_stream<7, EXACT, STORE>(c, warped, stats, act, st); return true; }
    if (rz == 6) { launch_stream<13, EXACT, STORE>(c, warped, stats, act, st); return true; }
    return false;
}

bool lncc_stream_supported(const LnccCtx &c)
{
    int rz = c.gt[2].radius;
    return (rz == 3 || rz == 6) && c.gt[0].radius <= 11 && c.gt[1].radius <= 11 && c.gt[0].radius > 0 && c.gt[1].radius > 0;
}

void lncc_value_fused(LnccCtx &c, const float *warped, double *out, const int *act, bool exact, cudaStream_t st)
{
    if (!exact && lnccf_value(c, warped, act, st)) {
        reduce_partials(c, c.nblk_used, 0, act, st);
        finalize_value(c, out, act, st);
        return;
    }
    if (exact) dispatch<true, false>(c, warped, c.bufB, act, st);
    else dispatch<false, false>(c, warped, c.bufB, act, st);
    reduce_partials(c, c.nblk_used, 0, act, st);
    finalize_value(c, out, act, st);
}
void lncc_stats_fused(LnccCtx &c, const float *warped, float *stats, const int *act, bool exact, cudaStream_t st)
{
    if (!exact && stats == c.bufB && lnccf_stats(c, warped, act, st)) return;
    if (exact) dispatch<true, true>(c, warped, stats, act, st);
    else dispatch<false, true>(c, warped, stats, act, st);
}

}  // namespace f3d
