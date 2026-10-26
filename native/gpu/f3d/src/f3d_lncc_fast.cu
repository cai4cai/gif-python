// SPDX-License-Identifier: BSD-3-Clause
// FAST-NUMERICS LNCC kernels (numerics = fast): same mathematical quantity as the exact kernels (Gaussian weighted local statistics,
// same truncated kernels, same local NCC and gradient formulas), but
//   * plain FP32 FMA accumulation (no error free two-sum emulation of NiftyReg's double accumulation),
//   * register blocked passes (x pass: 4 outputs per thread from a register window, y pass: all channels per column, z pass: register ring),
//     so that the shared memory pipe is not the limit,
//   * all six statistics channels in ONE kernel (no 3 + 3 split, no store/reload of intermediate statistics),
//   * epilogues in FP32 (the exact kernels use FP64 temporaries like NiftyReg).
// Three modes of the same kernel body:
//   MODE 0: local NCC value (reduction of |ncc| and count)
//   MODE 1: gradient temporaries t1,t2,t3 (+ smoothed density) for the gradient phase
//   MODE 2: smoothing of t1,t2,t3 fused with the per voxel gradient combination
// Requires: no reference mask (all voxels active), 16x16 tiles, radii of the specialised set.
#include "f3d_internal.h"
#include "f3d_lncc_common.cuh"
#include <cmath>
#include <cstring>

namespace f3d {

struct FTapArgs { float x[17]; float y[17]; float z[13]; };

#ifndef LF_MINB
#define LF_MINB 2
#endif

// statistics from the six smoothed channels, FP32 (reciprocal of the density once)
struct FStats { float mr, mw, sr, sw, corr; };
__device__ __forceinline__ FStats fstats_from(const float *o)
{
    float rd = __frcp_rn(o[0]);
    FStats s;
    s.mr = o[1] * rd;
    float qr = o[2] * rd;
    s.mw = o[3] * rd;
    float qw = o[4] * rd;
    s.corr = o[5] * rd;
    s.sr = sqrtf(fmaxf(__fmaf_rn(-s.mr, s.mr, qr), 0.f));
    s.sw = sqrtf(fmaxf(__fmaf_rn(-s.mw, s.mw, qw), 0.f));
    // NiftyReg: sdev < 1e-6 -> 0 (the local ncc is then non finite and the voxel is not counted)
    if (s.sr < 1.e-06f) s.sr = 0.f;
    if (s.sw < 1.e-06f) s.sw = 0.f;
    return s;
}

template <int RX, int RY, int RZ, int MODE>
__global__ void __launch_bounds__(256, LF_MINB)
    lnccf_k(const float *__restrict__ ref, const float *__restrict__ warped, size_t N, int nx, int ny, int nz, const FTapArgs ta,
            double *partial, int nblk, float *stats /*bufB: [B][6][N], ch 0 = density*/, float *gp /*bufA: temporaries ch 1..3*/,
            const float *__restrict__ wgrad, float *__restrict__ voxgrad, const double *__restrict__ acc, const int *act, int tilesX)
{
    constexpr int TW = 16, TH = 16, NT = 256, S = 2 * RZ + 1;
    constexpr int NC = (MODE == 2) ? 3 : 6;
    constexpr int WC = TW + 2 * RX, WR = TH + 2 * RY, WCP = WC + 1;
    constexpr int XB = 4, XG = TW / XB;
    constexpr int XPITCH = 34;   // 2 mod 32: the y pass (16 columns x 2 rows per warp) reads conflict free, the x pass writes rows consecutively
    static_assert(WR <= XPITCH, "pitch");
    __shared__ float inbuf[NC * WR * WCP];
    __shared__ float xbuf[NC * TW * XPITCH];
    const int b = blockIdx.y;
    if (act && !act[b]) return;
    const int tid = threadIdx.x;
    const int tx0 = (blockIdx.x % tilesX) * TW, ty0 = (blockIdx.x / tilesX) * TH;
    const float *w = warped + (size_t)b * N;
    const int cx = tid % TW, cy = tid / TW;
    const int xo = tx0 + cx, yo = ty0 + cy;
    const bool colIn = (xo < nx && yo < ny);
    float *gt = gp + (size_t)b * 6 * N;
    float *sb = stats + (size_t)b * 6 * N;
    const float *tp = gp + (size_t)b * 6 * N;   // MODE 2: temporaries t1..t3 at channels 1..3
    const float *dn = stats + (size_t)b * 6 * N;
    const float *wg = wgrad + (size_t)b * 3 * N;
    float *og = voxgrad + (size_t)b * 3 * N;

    // z range holding valid voxels of this tile
    __shared__ int szmin, szmax;
    if (tid == 0) { szmin = nz; szmax = -1; }
    __syncthreads();
    {
        int lo = nz, hi = -1;
        if (colIn) {
            for (int z = 0; z < nz; z++) {
                size_t id = ((size_t)z * ny + yo) * nx + xo;
                float r = ref[id], wv = w[id];
                if (r == r && wv == wv) { lo = z; break; }
            }
            if (lo < nz)
                for (int z = nz - 1; z >= lo; z--) {
                    size_t id = ((size_t)z * ny + yo) * nx + xo;
                    float r = ref[id], wv = w[id];
                    if (r == r && wv == wv) { hi = z; break; }
                }
        }
        if (lo < nz) { atomicMin(&szmin, lo); atomicMax(&szmax, hi); }
    }
    __syncthreads();
    const int zmin = szmin, zmax = szmax;
    if (MODE == 1 && colIn) {   // temporaries of voxels outside the processed z range (all invalid): zero
        for (int z = 0; z < nz; z++) {
            if (z >= zmin && z <= zmax) continue;
            size_t id = ((size_t)z * ny + yo) * nx + xo;
            gt[N + id] = 0.f; gt[2 * N + id] = 0.f; gt[3 * N + id] = 0.f;
        }
    }
    if (MODE == 2 && colIn) {
        for (int z = 0; z < nz; z++) {
            if (z >= zmin && z <= zmax) continue;
            size_t id = ((size_t)z * ny + yo) * nx + xo;
            og[id] = 0.f; og[N + id] = 0.f; og[2 * N + id] = 0.f;
        }
    }
    if (zmax < 0) {
        if (MODE != 2 && partial && tid == 0) {
            partial[((size_t)b * nblk + blockIdx.x) * 2] = 0.0;
            partial[((size_t)b * nblk + blockIdx.x) * 2 + 1] = 0.0;
        }
        return;
    }
    float adjusted = 0.f;
    if (MODE == 2) adjusted = (float)(1.0 / acc[(size_t)b * 4 + 2]);

    float ring[S][NC];
#pragma unroll
    for (int s = 0; s < S; s++)
#pragma unroll
        for (int c = 0; c < NC; c++) ring[s][c] = 0.f;
    double dsum = 0.0;
    int icnt = 0;

    constexpr int NE = (WR * WC + NT - 1) / NT;
    float pa[NE], pb[NE], pc[NE];
    auto fetch = [&](int zq) {
        const size_t pl = (size_t)zq * nx * ny;
#pragma unroll
        for (int e = 0; e < NE; e++) {
            int i = tid + e * NT;
            if (MODE == 2) { pa[e] = 0.f; pb[e] = 0.f; pc[e] = 0.f; }
            else { pa[e] = nanf(""); pb[e] = 0.f; }
            if (zq >= 0 && zq < nz && i < WR * WC) {
                int row = i / WC, col = i - row * WC;
                int x = tx0 - RX + col, y = ty0 - RY + row;
                if (x >= 0 && x < nx && y >= 0 && y < ny) {
                    size_t id = pl + (size_t)y * nx + x;
                    if (MODE == 2) { pa[e] = tp[N + id]; pb[e] = tp[2 * N + id]; pc[e] = tp[3 * N + id]; }
                    else { pa[e] = w[id]; pb[e] = ref[id]; }
                }
            }
        }
    };
    fetch(zmin - RZ);
    unsigned vmask = 0;   // bit k: valid voxel of this column k planes ago

    for (int zi = zmin - RZ; zi <= zmax + RZ; ++zi) {
        float nw[NC];
        const int zo = zi - RZ;
        const bool doOut = colIn && zo >= zmin && zo <= zmax;
        const size_t eid = ((size_t)zo * ny + yo) * nx + xo;
        if (zi >= 0 && zi < nz) {
#pragma unroll
            for (int e = 0; e < NE; e++) {
                int i = tid + e * NT;
                if (i < WR * WC) {
                    int row = i / WC, col = i - row * WC;
                    float *p = inbuf + row * WCP + col;
                    if (MODE == 2) {
                        p[0] = pa[e]; p[WR * WCP] = pb[e]; p[2 * WR * WCP] = pc[e];
                    } else {
                        float wv = pa[e], r = pb[e];
                        float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, c4 = 0.f, c5 = 0.f;
                        if ((r == r) && (wv == wv)) { c0 = 1.f; c1 = r; c2 = r * r; c3 = wv; c4 = wv * wv; c5 = r * wv; }
                        p[0] = c0; p[WR * WCP] = c1; p[2 * WR * WCP] = c2; p[3 * WR * WCP] = c3; p[4 * WR * WCP] = c4; p[5 * WR * WCP] = c5;
                    }
                }
            }
            fetch(zi + 1);
            __syncthreads();
            if (MODE != 2) vmask = (vmask << 1) | (inbuf[(RY + cy) * WCP + RX + cx] != 0.f ? 1u : 0u);
            // ---- x pass: item = (channel, 4 output columns, row); rows fastest so that a warp reads/writes consecutive rows
            {
                constexpr int XTOT = NC * XG * WR;
#pragma unroll 1
                for (int it = tid; it < XTOT; it += NT) {
                    const int row = it % WR, rest = it / WR;
                    const int xg = rest % XG, c = rest / XG;
                    const float *pp = inbuf + (c * WR + row) * WCP + xg * XB;
                    float win[XB + 2 * RX];
#pragma unroll
                    for (int j = 0; j < XB + 2 * RX; j++) win[j] = pp[j];
                    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
                    for (int k = 0; k <= 2 * RX; k++) {
                        const float wk = ta.x[k];
                        a0 = __fmaf_rn(wk, win[k], a0);
                        a1 = __fmaf_rn(wk, win[k + 1], a1);
                        a2 = __fmaf_rn(wk, win[k + 2], a2);
                        a3 = __fmaf_rn(wk, win[k + 3], a3);
                    }
                    float *xp = xbuf + (c * TW + xg * XB) * XPITCH + row;
                    xp[0] = a0; xp[XPITCH] = a1; xp[2 * XPITCH] = a2; xp[3 * XPITCH] = a3;
                }
            }
            __syncthreads();
            // ---- y pass (all channels of the column)
            {
                float acc6[NC];
#pragma unroll
                for (int c = 0; c < NC; c++) acc6[c] = 0.f;
                const float *p = xbuf + cx * XPITCH + cy;
#pragma unroll
                for (int k = 0; k <= 2 * RY; k++) {
                    const float wk = ta.y[k];
#pragma unroll
                    for (int c = 0; c < NC; c++) acc6[c] = __fmaf_rn(wk, p[c * TW * XPITCH + k], acc6[c]);
                }
#pragma unroll
                for (int c = 0; c < NC; c++) nw[c] = acc6[c];
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
#pragma unroll
            for (int c = 0; c < NC; c++) o[c] = 0.f;
#pragma unroll
            for (int k = 0; k < S; k++) {
                const float wk = ta.z[k];
#pragma unroll
                for (int c = 0; c < NC; c++) o[c] = __fmaf_rn(wk, ring[k][c], o[c]);
            }
            if (doOut) {
                if (MODE == 0) {
                    if ((vmask >> RZ) & 1u) {
                        FStats s = fstats_from(o);
                        float lv = (s.corr - (s.mr * s.mw)) / (s.sr * s.sw);
                        if (lv == lv && !isinf(lv)) { dsum += (double)fabsf(lv); icnt++; }
                    }
                } else if (MODE == 1) {
                    float t1 = 0.f, t2 = 0.f, t3 = 0.f;
                    if ((vmask >> RZ) & 1u) {
                        FStats s = fstats_from(o);
                        float correla = s.corr - s.mr * s.mw;
                        float inv = __frcp_rn(s.sr * s.sw);
                        float u1 = inv;
                        float u2 = correla * inv * __frcp_rn(s.sw * s.sw);
                        float u3 = __fmaf_rn(s.mw, u2, -s.mr * u1);
                        if (u1 == u1 && !isinf(u1) && u2 == u2 && !isinf(u2) && u3 == u3 && !isinf(u3)) {
                            if (correla < 0.f) { u1 = -u1; u2 = -u2; u3 = -u3; }
                            t1 = u1; t2 = u2; t3 = u3; icnt++;
                        }
                        sb[eid] = o[0];
                    }
                    gt[N + eid] = t1; gt[2 * N + eid] = t2; gt[3 * N + eid] = t3;
                } else {
                    float g0 = 0.f, g1 = 0.f, g2 = 0.f;
                    const float er = ref[eid], ew = w[eid];
                    if ((er == er) && (ew == ew)) {
                        float rd = __frcp_rn(dn[eid]);
                        float cm = o[0] * rd * er - o[1] * rd * ew + o[2] * rd;
                        float common = cm * adjusted;
                        g0 = -wg[eid] * common;
                        g1 = -wg[N + eid] * common;
                        g2 = -wg[2 * N + eid] * common;
                        if (g0 != g0 || isinf(g0)) g0 = 0.f;
                        if (g1 != g1 || isinf(g1)) g1 = 0.f;
                        if (g2 != g2 || isinf(g2)) g2 = 0.f;
                    }
                    og[eid] = g0; og[N + eid] = g1; og[2 * N + eid] = g2;
                }
            }
        }
    }
    if (MODE != 2) {
        __shared__ double shd[32];
        double dc = (double)icnt;
        dsum = block_sum_d(dsum, shd);
        dc = block_sum_d(dc, shd);
        if (tid == 0 && partial) {
            if (MODE == 0) {
                partial[((size_t)b * nblk + blockIdx.x) * 2] = dsum;
                partial[((size_t)b * nblk + blockIdx.x) * 2 + 1] = dc;
            } else {
                partial[((size_t)b * nblk + blockIdx.x) * 2] = dc;
                partial[((size_t)b * nblk + blockIdx.x) * 2 + 1] = 0.0;
            }
        }
    }
}


// ======================================================================================================================
// Version 2: shared memory traffic reduced by register blocking of BOTH smooth passes, vector loads, no per plane index arithmetic.
//   x pass: item = (channel, 4 output columns, row): 20 inputs as 5 LDS.128 -> 4 outputs, stored transposed xb[c][col][row] (pitch 48)
//   y pass: item = (channel, column, 4 output rows): 20 inputs as 5 LDS.128 (consecutive rows of the transposed buffer) -> 4 outputs -> yb
//   z pass: the thread owning the column reads its NC values from yb, register ring as before.
template <int RX, int RY, int RZ, int MODE>
__global__ void __launch_bounds__(256, LF_MINB)
    lnccf2_k(const float *__restrict__ ref, const float *__restrict__ warped, size_t N, int nx, int ny, int nz, const FTapArgs ta,
             double *partial, int nblk, float *stats, float *gp, const float *__restrict__ wgrad, float *__restrict__ voxgrad,
             const double *__restrict__ acc, const int *act, int tilesX)
{
    constexpr int TW = 16, TH = 16, NT = 256, S = 2 * RZ + 1;
    constexpr int NC = (MODE == 2) ? 3 : 6;
    constexpr int WC = TW + 2 * RX, WR = TH + 2 * RY;
    constexpr int IP4 = ((WC / 4) % 2 == 0) ? WC / 4 + 1 : WC / 4, IP = IP4 * 4;   // inbuf row pitch (odd number of float4)
    constexpr int XB = 4, XG = TW / XB, XP = 48, YP = 34;
    constexpr int XWIN = XB + 2 * RX, XNV = (XWIN + 3) / 4;       // x window floats / float4 loads
    constexpr int YWIN = 4 + 2 * RY, YNV = (YWIN + 3) / 4;
    static_assert(WC % 4 == 0 && WR <= XP, "layout");
    constexpr int INSZ = (NC * WR * IP > NC * TW * YP) ? NC * WR * IP : NC * TW * YP;
    __shared__ __align__(16) float inbuf[INSZ];     // window channels; after the x pass it is reused for the y pass output
    __shared__ __align__(16) float xb[NC * TW * XP];
    float *yb = inbuf;
    const int b = blockIdx.y;
    if (act && !act[b]) return;
    const int tid = threadIdx.x;
    const int tx0 = (blockIdx.x % tilesX) * TW, ty0 = (blockIdx.x / tilesX) * TH;
    const float *w = warped + (size_t)b * N;
    const int cx = tid % TW, cy = tid / TW;
    const int xo = tx0 + cx, yo = ty0 + cy;
    const bool colIn = (xo < nx && yo < ny);
    float *gt = gp + (size_t)b * 6 * N;
    float *sb = stats + (size_t)b * 6 * N;
    const float *tp = gp + (size_t)b * 6 * N;
    const float *dn = stats + (size_t)b * 6 * N;
    const float *wg = wgrad + (size_t)b * 3 * N;
    float *og = voxgrad + (size_t)b * 3 * N;

    __shared__ int szmin, szmax;
    if (tid == 0) { szmin = nz; szmax = -1; }
    __syncthreads();
    {
        int lo = nz, hi = -1;
        if (colIn) {
            for (int z = 0; z < nz; z++) {
                size_t id = ((size_t)z * ny + yo) * nx + xo;
                float r = ref[id], wv = w[id];
                if (r == r && wv == wv) { lo = z; break; }
            }
            if (lo < nz)
                for (int z = nz - 1; z >= lo; z--) {
                    size_t id = ((size_t)z * ny + yo) * nx + xo;
                    float r = ref[id], wv = w[id];
                    if (r == r && wv == wv) { hi = z; break; }
                }
        }
        if (lo < nz) { atomicMin(&szmin, lo); atomicMax(&szmax, hi); }
    }
    __syncthreads();
    const int zmin = szmin, zmax = szmax;
    if (MODE == 1 && colIn) {
        for (int z = 0; z < nz; z++) {
            if (z >= zmin && z <= zmax) continue;
            size_t id = ((size_t)z * ny + yo) * nx + xo;
            gt[N + id] = 0.f; gt[2 * N + id] = 0.f; gt[3 * N + id] = 0.f;
        }
    }
    if (MODE == 2 && colIn) {
        for (int z = 0; z < nz; z++) {
            if (z >= zmin && z <= zmax) continue;
            size_t id = ((size_t)z * ny + yo) * nx + xo;
            og[id] = 0.f; og[N + id] = 0.f; og[2 * N + id] = 0.f;
        }
    }
    if (zmax < 0) {
        if (MODE != 2 && partial && tid == 0) {
            partial[((size_t)b * nblk + blockIdx.x) * 2] = 0.0;
            partial[((size_t)b * nblk + blockIdx.x) * 2 + 1] = 0.0;
        }
        return;
    }
    float adjusted = 0.f;
    if (MODE == 2) adjusted = (float)(1.0 / acc[(size_t)b * 4 + 2]);

    float ring[S][NC];
#pragma unroll
    for (int s = 0; s < S; s++)
#pragma unroll
        for (int c = 0; c < NC; c++) ring[s][c] = 0.f;
    double dsum = 0.0;
    int icnt = 0;

    // staging of the window: element e of this thread = window voxel i = tid + e*NT; source offset inside a plane (or -1), smem offset
    constexpr int NE = (WR * WC + NT - 1) / NT;
    int soff[NE], doff[NE];
#pragma unroll
    for (int e = 0; e < NE; e++) {
        int i = tid + e * NT;
        soff[e] = -1; doff[e] = 0;
        if (i < WR * WC) {
            int row = i / WC, col = i - row * WC;
            int x = tx0 - RX + col, y = ty0 - RY + row;
            doff[e] = row * IP + col;
            if (x >= 0 && x < nx && y >= 0 && y < ny) soff[e] = y * nx + x;
        }
    }
    float pa[NE], pb[NE], pc[NE];
    auto fetch = [&](int zq) {
        const size_t pl = (size_t)zq * nx * ny;
#pragma unroll
        for (int e = 0; e < NE; e++) {
            if (MODE == 2) { pa[e] = 0.f; pb[e] = 0.f; pc[e] = 0.f; }
            else { pa[e] = nanf(""); pb[e] = 0.f; }
            if (zq >= 0 && zq < nz && soff[e] >= 0) {
                size_t id = pl + soff[e];
                if (MODE == 2) { pa[e] = tp[N + id]; pb[e] = tp[2 * N + id]; pc[e] = tp[3 * N + id]; }
                else { pa[e] = w[id]; pb[e] = ref[id]; }
            }
        }
    };
    fetch(zmin - RZ);
    unsigned vmask = 0;

    for (int zb = zmin - RZ; zb <= zmax + RZ; zb += S) {
#pragma unroll
    for (int j = 0; j < S; j++) {
        const int zi = zb + j;
        if (zi > zmax + RZ) break;
        float nw[NC];
        const int zo = zi - RZ;
        const bool doOut = colIn && zo >= zmin && zo <= zmax;
        const size_t eid = ((size_t)zo * ny + yo) * nx + xo;
        if (zi >= 0 && zi < nz) {
#pragma unroll
            for (int e = 0; e < NE; e++) {
                if (tid + e * NT < WR * WC) {
                    float *p = inbuf + doff[e];
                    if (MODE == 2) {
                        p[0] = pa[e]; p[WR * IP] = pb[e]; p[2 * WR * IP] = pc[e];
                    } else {
                        float wv = pa[e], r = pb[e];
                        float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, c4 = 0.f, c5 = 0.f;
                        if ((r == r) && (wv == wv)) { c0 = 1.f; c1 = r; c2 = r * r; c3 = wv; c4 = wv * wv; c5 = r * wv; }
                        p[0] = c0; p[WR * IP] = c1; p[2 * WR * IP] = c2; p[3 * WR * IP] = c3; p[4 * WR * IP] = c4; p[5 * WR * IP] = c5;
                    }
                }
            }
            fetch(zi + 1);
            __syncthreads();
            if (MODE != 2) vmask = (vmask << 1) | (inbuf[(RY + cy) * IP + RX + cx] != 0.f ? 1u : 0u);
            // ---- x pass
            {
                constexpr int XTOT = NC * XG * WR;
#pragma unroll 1
                for (int it = tid; it < XTOT; it += NT) {
                    const int row = it % WR, rest = it / WR;
                    const int xg = rest % XG, c = rest / XG;
                    const float4 *pp = reinterpret_cast<const float4 *>(inbuf + (c * WR + row) * IP + xg * XB);
                    float win[XNV * 4];
#pragma unroll
                    for (int v = 0; v < XNV; v++) { float4 q = pp[v]; win[4 * v] = q.x; win[4 * v + 1] = q.y; win[4 * v + 2] = q.z; win[4 * v + 3] = q.w; }
                    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
                    for (int k = 0; k <= 2 * RX; k++) {
                        const float wk = ta.x[k];
                        a0 = __fmaf_rn(wk, win[k], a0);
                        a1 = __fmaf_rn(wk, win[k + 1], a1);
                        a2 = __fmaf_rn(wk, win[k + 2], a2);
                        a3 = __fmaf_rn(wk, win[k + 3], a3);
                    }
                    float *xp = xb + (c * TW + xg * XB) * XP + row;
                    xp[0] = a0; xp[XP] = a1; xp[2 * XP] = a2; xp[3 * XP] = a3;
                }
            }
            __syncthreads();
            // ---- y pass: item = (c, cx, yg): lanes: yg fastest, then cx
            {
                constexpr int YTOT = NC * TW * 4;
#pragma unroll 1
                for (int it = tid; it < YTOT; it += NT) {
                    const int yg = it & 3, cxx = (it >> 2) & 15, c = it >> 6;
                    const float4 *pp = reinterpret_cast<const float4 *>(xb + (c * TW + cxx) * XP + yg * 4);
                    float win[YNV * 4];
#pragma unroll
                    for (int v = 0; v < YNV; v++) { float4 q = pp[v]; win[4 * v] = q.x; win[4 * v + 1] = q.y; win[4 * v + 2] = q.z; win[4 * v + 3] = q.w; }
                    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
                    for (int k = 0; k <= 2 * RY; k++) {
                        const float wk = ta.y[k];
                        a0 = __fmaf_rn(wk, win[k], a0);
                        a1 = __fmaf_rn(wk, win[k + 1], a1);
                        a2 = __fmaf_rn(wk, win[k + 2], a2);
                        a3 = __fmaf_rn(wk, win[k + 3], a3);
                    }
                    float *yp = yb + (c * TW + cxx) * YP + yg * 4;
                    yp[0] = a0; yp[1] = a1; yp[2] = a2; yp[3] = a3;
                }
            }
            __syncthreads();
#pragma unroll
            for (int c = 0; c < NC; c++) nw[c] = yb[(c * TW + cx) * YP + cy];
            __syncthreads();   // yb aliases the window buffer that the next plane stages
        } else {
            fetch(zi + 1);
            vmask <<= 1;
#pragma unroll
            for (int c = 0; c < NC; c++) nw[c] = 0.f;
        }
#pragma unroll
        for (int c = 0; c < NC; c++) ring[j][c] = nw[c];
        if (zo >= zmin && zo <= zmax) {
            float o[NC];
#pragma unroll
            for (int c = 0; c < NC; c++) o[c] = 0.f;
#pragma unroll
            for (int k = 0; k < S; k++) {
                const float wk = ta.z[k];
#pragma unroll
                for (int c = 0; c < NC; c++) o[c] = __fmaf_rn(wk, ring[(j + 1 + k) % S][c], o[c]);
            }
            if (doOut) {
                if (MODE == 0) {
                    if ((vmask >> RZ) & 1u) {
                        FStats s = fstats_from(o);
                        float lv = (s.corr - (s.mr * s.mw)) / (s.sr * s.sw);
                        if (lv == lv && !isinf(lv)) { dsum += (double)fabsf(lv); icnt++; }
                    }
                } else if (MODE == 1) {
                    float t1 = 0.f, t2 = 0.f, t3 = 0.f;
                    if ((vmask >> RZ) & 1u) {
                        FStats s = fstats_from(o);
                        float correla = s.corr - s.mr * s.mw;
                        float inv = __frcp_rn(s.sr * s.sw);
                        float u1 = inv;
                        float u2 = correla * inv * __frcp_rn(s.sw * s.sw);
                        float u3 = __fmaf_rn(s.mw, u2, -s.mr * u1);
                        if (u1 == u1 && !isinf(u1) && u2 == u2 && !isinf(u2) && u3 == u3 && !isinf(u3)) {
                            if (correla < 0.f) { u1 = -u1; u2 = -u2; u3 = -u3; }
                            t1 = u1; t2 = u2; t3 = u3; icnt++;
                        }
                        sb[eid] = o[0];
                    }
                    gt[N + eid] = t1; gt[2 * N + eid] = t2; gt[3 * N + eid] = t3;
                } else {
                    float g0 = 0.f, g1 = 0.f, g2 = 0.f;
                    const float er = ref[eid], ew = w[eid];
                    if ((er == er) && (ew == ew)) {
                        float rd = __frcp_rn(dn[eid]);
                        float cm = o[0] * rd * er - o[1] * rd * ew + o[2] * rd;
                        float common = cm * adjusted;
                        g0 = -wg[eid] * common;
                        g1 = -wg[N + eid] * common;
                        g2 = -wg[2 * N + eid] * common;
                        if (g0 != g0 || isinf(g0)) g0 = 0.f;
                        if (g1 != g1 || isinf(g1)) g1 = 0.f;
                        if (g2 != g2 || isinf(g2)) g2 = 0.f;
                    }
                    og[eid] = g0; og[N + eid] = g1; og[2 * N + eid] = g2;
                }
            }
        }
    }
    }
    if (MODE != 2) {
        __shared__ double shd[32];
        double dc = (double)icnt;
        dsum = block_sum_d(dsum, shd);
        dc = block_sum_d(dc, shd);
        if (tid == 0 && partial) {
            if (MODE == 0) {
                partial[((size_t)b * nblk + blockIdx.x) * 2] = dsum;
                partial[((size_t)b * nblk + blockIdx.x) * 2 + 1] = dc;
            } else {
                partial[((size_t)b * nblk + blockIdx.x) * 2] = dc;
                partial[((size_t)b * nblk + blockIdx.x) * 2 + 1] = 0.0;
            }
        }
    }
}

template <int RX, int RY, int RZ, int MODE>
static void launch_f(LnccCtx &c, const float *warped, const float *wgrad, float *voxgrad, const int *act, cudaStream_t st)
{
    FTapArgs ta;
    memset(&ta, 0, sizeof ta);
    for (int k = 0; k <= 2 * RX; k++) ta.x[k] = c.gt[0].h[k];
    for (int k = 0; k <= 2 * RY; k++) ta.y[k] = c.gt[1].h[k];
    for (int k = 0; k <= 2 * RZ; k++) ta.z[k] = c.gt[2].h[k];
    int tilesX = cdiv(c.dims[0], 16), tilesY = cdiv(c.dims[1], 16);
    if (c.lfver == 1)
        lnccf_k<RX, RY, RZ, MODE><<<dim3(tilesX * tilesY, c.B), 256, 0, st>>>(c.ref, warped, c.N, c.dims[0], c.dims[1], c.dims[2], ta, c.partial,
                                                                           tilesX * tilesY, c.bufB, c.bufA, wgrad, voxgrad, c.acc, act, tilesX);
    else
        lnccf2_k<RX, RY, RZ, MODE><<<dim3(tilesX * tilesY, c.B), 256, 0, st>>>(c.ref, warped, c.N, c.dims[0], c.dims[1], c.dims[2], ta, c.partial,
                                                                            tilesX * tilesY, c.bufB, c.bufA, wgrad, voxgrad, c.acc, act, tilesX);
    if (MODE != 2) c.nblk_used = tilesX * tilesY;
}

template <int MODE>
static bool dispatch_f(LnccCtx &c, const float *warped, const float *wgrad, float *voxgrad, const int *act, cudaStream_t st)
{
    static const bool off = getenv("F3D_NOFASTK") != nullptr;
    if (off || c.mask) return false;
    int rx = c.gt[0].radius, ry = c.gt[1].radius, rz = c.gt[2].radius;
#define DF(X, Y, Z) if (rx == X && ry == Y && rz == Z) { launch_f<X, Y, Z, MODE>(c, warped, wgrad, voxgrad, act, st); return true; }
    DF(8, 8, 6) DF(8, 7, 6) DF(4, 4, 3) DF(4, 3, 3) DF(2, 2, 3) DF(2, 1, 3)
#undef DF
    return false;
}

bool lnccf_value(LnccCtx &c, const float *warped, const int *act, cudaStream_t st) { return dispatch_f<0>(c, warped, nullptr, nullptr, act, st); }
bool lnccf_stats(LnccCtx &c, const float *warped, const int *act, cudaStream_t st)
{
    bool ok = dispatch_f<1>(c, warped, nullptr, nullptr, act, st);
    if (ok) c.gp_done = true;
    return ok;
}
bool lnccf_smooth(LnccCtx &c, const float *warped, const float *wgrad, float *voxgrad, const int *act, cudaStream_t st)
{
    return dispatch_f<2>(c, warped, wgrad, voxgrad, act, st);
}

}  // namespace f3d
