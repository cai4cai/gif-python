// SPDX-License-Identifier: BSD-3-Clause
// Control point grid helpers, B-spline deformation + linear resampling (+ image gradient).
#include "f3d_internal.h"
#include "f3d_refine_gen.cuh"
#include <cmath>
#include <algorithm>

namespace f3d {

// ------------------------------------------------------------------ host: basis tables (NiftyReg get_BSplineBasisValues<float>)
static void bspline_basis(float basis, float v[4])
{
    float FF = basis * basis;
    float FFF = FF * basis;
    float MF = (float)(1.0 - basis);
    v[0] = (float)((double)(MF * MF * MF) / 6.0);
    v[1] = (float)((3.0 * FFF - 6.0 * FF + 4.0) / 6.0);
    v[2] = (float)((-3.0 * FFF + 3.0 * FF + 3.0 * basis + 1.0) / 6.0);
    v[3] = (float)(FFF / 6.0);
}

std::vector<AxisTab> build_tab_generic(int n, float gvs)
{
    std::vector<AxisTab> t(n);
    for (int i = 0; i < n; i++) {
        int pre = (int)((float)i / gvs);
        float basis = (float)i / gvs - (float)pre;
        if (basis < 0.f) basis = 0.f;
        t[i].pre = pre;
        bspline_basis(basis, t[i].w);
    }
    return t;
}
std::vector<AxisTab> build_tab_lut(int n, int nnodes)
{
    // NiftyReg: loops over cells [0,nnodes-3), voxel = cell*5 + local, local in [0,5). basis=(float)local/5.
    float lw[5][4];
    for (int l = 0; l < 5; l++) {
        float basis = (float)((double)l / 5.);
        bspline_basis(basis, lw[l]);
    }
    std::vector<AxisTab> t(n);
    for (int i = 0; i < n; i++) {
        int cell = i / 5, l = i % 5;
        if (cell < nnodes - 3) {
            t[i].pre = cell;
            for (int k = 0; k < 4; k++) t[i].w[k] = lw[l][k];
        } else {
            t[i].pre = -1;
            for (int k = 0; k < 4; k++) t[i].w[k] = 0.f;
        }
    }
    return t;
}

// ------------------------------------------------------------------ grid init from an affine (reg_affine_getDeformationField on the grid)
std::vector<float> grid_init_affine(const GridGeom &g, const M44 &aff)
{
    size_t nn = g.nnodes();
    std::vector<float> o(3 * nn);
    M44 T = m44_mul(aff, g.xyz);
    for (int z = 0; z < g.n[2]; z++)
        for (int y = 0; y < g.n[1]; y++)
            for (int x = 0; x < g.n[0]; x++) {
                double v[3] = {(double)x, (double)y, (double)z};
                double p[3];
                for (int i = 0; i < 3; i++)
                    p[i] = (double)T.m[i][0] * v[0] + (double)T.m[i][1] * v[1] + (double)T.m[i][2] * v[2] + (double)T.m[i][3];
                size_t id = ((size_t)z * g.n[1] + y) * g.n[0] + x;
                o[id] = (float)p[0]; o[nn + id] = (float)p[1]; o[2 * nn + id] = (float)p[2];
            }
    return o;
}

// ------------------------------------------------------------------ grid refinement
__global__ void refine_k(const float *oldg, float *newg, int ox, int oy, int oz, int nx, int ny, int nz, size_t oStride,
                         size_t nStride)
{
    size_t onn = (size_t)ox * oy * oz, nnn = (size_t)nx * ny * nz;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= onn) return;
    int comp = blockIdx.z, b = blockIdx.y;
    const float *og = oldg + (size_t)b * oStride + (size_t)comp * onn;
    float *ng = newg + (size_t)b * nStride + (size_t)comp * nnn;
    int x = i % ox, y = (i / ox) % oy, z = i / ((size_t)ox * oy);
    int X = 2 * x - 1, Y = 2 * y - 1, Z = 2 * z - 1;
    if (Z >= nz || Y >= ny || X >= nx) return;
    float o[8];
    refine_octants(og, ox, oy, oz, x, y, z, o);
#pragma unroll
    for (int k = 0; k < 8; k++) {
        int XX = X + (k & 1), YY = Y + ((k >> 1) & 1), ZZ = Z + ((k >> 2) & 1);
        if (XX >= 0 && XX < nx && YY >= 0 && YY < ny && ZZ >= 0 && ZZ < nz) ng[((size_t)ZZ * ny + YY) * nx + XX] = o[k];
    }
}
void grid_refine(const float *d_old, const GridGeom &go, float *d_new, const GridGeom &gn, int B, cudaStream_t st)
{
    size_t onn = go.nnodes(), nnn = gn.nnodes();
    CK(cudaMemsetAsync(d_new, 0, (size_t)B * 3 * nnn * sizeof(float), st));   // nodes that are never set stay 0 (NiftyReg calloc)
    dim3 grid(cdiv(onn, 128), B, 3);
    refine_k<<<grid, 128, 0, st>>>(d_old, d_new, go.n[0], go.n[1], go.n[2], gn.n[0], gn.n[1], gn.n[2], 3 * onn, 3 * nnn);
}


// ------------------------------------------------------------------ double-float helpers (the GPU has very slow fp64)
struct DF { float hi, lo; };
__device__ __forceinline__ DF two_sum(float a, float b)
{
    float s = a + b, bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}
__device__ __forceinline__ DF quick_two_sum(float a, float b)   // |a| >= |b|
{
    float s = a + b;
    return {s, b - (s - a)};
}
__device__ __forceinline__ DF two_prod(float a, float b)
{
    float p = __fmul_rn(a, b);
    return {p, __fmaf_rn(a, b, -p)};
}
__device__ __forceinline__ DF df_add(DF x, DF y)
{
    DF s = two_sum(x.hi, y.hi);
    DF t = two_sum(x.lo, y.lo);
    float lo = s.lo + t.hi;
    DF u = quick_two_sum(s.hi, lo);
    lo = t.lo + u.lo;
    return quick_two_sum(u.hi, lo);
}
__device__ __forceinline__ DF df_mul(DF x, DF y)
{
    DF p = two_prod(x.hi, y.hi);
    p.lo += x.hi * y.lo + x.lo * y.hi;
    return quick_two_sum(p.hi, p.lo);
}
__device__ __forceinline__ DF df_mul_f(float x, DF y)
{
    DF p = two_prod(x, y.hi);
    p.lo = __fmaf_rn(x, y.lo, p.lo);
    return quick_two_sum(p.hi, p.lo);
}

// ------------------------------------------------------------------ deformation + resampling
#define WARP_XN 32
#define WARP_YS 6
// ROW mode: a block covers `rpb` consecutive rows of `bx` consecutive voxels (bx * rpb = 128) of one slice; the control point
// neighbourhood of the whole block (4 slices x <=WARP_YS rows x <=WARP_XN nodes, 3 components as float4) is staged in shared memory once,
// so the 64-node loop of every voxel reads shared memory (same nodes, same order of operations -> identical results).
template <bool DP, bool ROW>
__global__ void __launch_bounds__(128)
    warp_k(WarpArgs a, const float *__restrict__ cpp, size_t cppStride, const float *__restrict__ flo,
                       size_t floStride, float *warped, float *wgrad, float *def, const int *act)
{
    size_t N = (size_t)a.rn[0] * a.rn[1] * a.rn[2];
    int b = blockIdx.y;
    if (act && !act[b]) return;
    size_t i;
    int x, y, z;
    bool inb = true;
    __shared__ float4 sm[ROW ? 4 * WARP_YS * WARP_XN : 1];
    __shared__ int sb_[4];
    int xlo = 0, ylo = 0;
    if (ROW) {
        const int bx = a.bx, nxb = (a.rn[0] + bx - 1) / bx, rpb = 128 / bx, nyg = (a.rn[1] + rpb - 1) / rpb;
        int blk = blockIdx.x;
        int xb = blk % nxb; int t = blk / nxb; int yg = t % nyg; z = t / nyg;
        x = xb * bx + (threadIdx.x & (bx - 1));
        y = yg * rpb + (threadIdx.x / bx);
        inb = (x < a.rn[0] && y < a.rn[1]);
        i = ((size_t)z * a.rn[1] + y) * a.rn[0] + x;
    } else {
        i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;
        x = i % a.rn[0]; y = (i / a.rn[0]) % a.rn[1]; z = i / ((size_t)a.rn[0] * a.rn[1]);
    }
    size_t nn = (size_t)a.gn[0] * a.gn[1] * a.gn[2];
    const float *cp = cpp + (size_t)b * cppStride;
    const float *fl = flo + (size_t)b * floStride;

    float world[3] = {0.f, 0.f, 0.f};
    bool active = inb && ((a.mask == nullptr) || (a.mask[i] > -1));
    AxisTab tx, ty, tz;
    tx.pre = ty.pre = tz.pre = -1;
    if (inb) { tx = a.tab[0][x]; ty = a.tab[1][y]; tz = a.tab[2][z]; }
    if (!ROW && a.cflag && tx.pre >= 0 && ty.pre >= 0 && tz.pre >= 0 &&
        a.cflag[(size_t)b * a.ncell + ((size_t)tz.pre * a.cn[1] + ty.pre) * a.cn[0] + tx.pre]) {
        // the whole cell maps outside the floating image: NaN value, zero image gradient (what the full computation yields)
        warped[(size_t)b * N + i] = nanf("");
        if (wgrad) {
            float *g = wgrad + (size_t)b * 3 * N;
            g[i] = 0.f; g[N + i] = 0.f; g[2 * N + i] = 0.f;
        }
        return;
    }
    if (ROW) {
        // block extent of the node neighbourhood
        if (threadIdx.x == 0) { sb_[0] = 1 << 30; sb_[1] = -1; sb_[2] = 1 << 30; sb_[3] = -1; }
        __syncthreads();
        if (inb && tx.pre >= 0 && ty.pre >= 0 && tz.pre >= 0) {
            atomicMin(&sb_[0], tx.pre); atomicMax(&sb_[1], tx.pre);
            atomicMin(&sb_[2], ty.pre); atomicMax(&sb_[3], ty.pre);
        }
        __syncthreads();
        if (sb_[1] >= 0) {
            xlo = sb_[0]; ylo = sb_[2];
            const int xhi = min(sb_[1] + 3, a.gn[0] - 1), yhi = min(sb_[3] + 3, a.gn[1] - 1);
            const int xc = xhi - xlo + 1, yc = yhi - ylo + 1;
            // slice index of the block: tz.pre of any in-range thread (uniform over the block)
            int zpre = a.tab[2][z].pre;
            for (int t = threadIdx.x; t < 4 * yc * xc; t += 128) {
                int c = t / (yc * xc), rem = t - c * (yc * xc);
                int yy = rem / xc, xx = rem - yy * xc;
                int Z = min(max(zpre + c, 0), a.gn[2] - 1);
                size_t id = ((size_t)Z * a.gn[1] + (ylo + yy)) * a.gn[0] + (xlo + xx);
                sm[(c * WARP_YS + yy) * WARP_XN + xx] = make_float4(cp[id], cp[nn + id], cp[2 * nn + id], 0.f);
            }
        }
        __syncthreads();
        if (!inb) return;
    }
    if (active) {
        if (tx.pre >= 0 && ty.pre >= 0 && tz.pre >= 0) {
            float rx = 0.f, ry = 0.f, rz = 0.f;
            if (ROW) {
                int Xn[4];
#pragma unroll
                for (int aa = 0; aa < 4; aa++) Xn[aa] = min(max(tx.pre + aa, 0), a.gn[0] - 1) - xlo;
#pragma unroll
                for (int c = 0; c < 4; c++) {
#pragma unroll
                    for (int bb = 0; bb < 4; bb++) {
                        int Y = min(max(ty.pre + bb, 0), a.gn[1] - 1) - ylo;
                        float yz = ty.w[bb] * tz.w[c];
                        const float4 *row = sm + (c * WARP_YS + Y) * WARP_XN;
#pragma unroll
                        for (int aa = 0; aa < 4; aa++) {
                            float4 v = row[Xn[aa]];
                            float w = tx.w[aa] * yz;
                            rx += v.x * w;
                            ry += v.y * w;
                            rz += v.z * w;
                        }
                    }
                }
            } else if (tx.pre + 3 <= a.gn[0] - 1 && ty.pre + 3 <= a.gn[1] - 1 && tz.pre + 3 <= a.gn[2] - 1) {
                // all 64 nodes inside the grid (no clamping): constant offsets from three base pointers
                const size_t S1 = a.gn[0], S2 = (size_t)a.gn[0] * a.gn[1];
                const float *p0 = cp + ((size_t)tz.pre * a.gn[1] + ty.pre) * a.gn[0] + tx.pre;
#pragma unroll
                for (int c = 0; c < 4; c++) {
#pragma unroll
                    for (int bb = 0; bb < 4; bb++) {
                        const float *qx = p0 + c * S2 + bb * S1;
                        const float *qy = qx + nn, *qz = qx + 2 * nn;
                        float yz = ty.w[bb] * tz.w[c];
#pragma unroll
                        for (int aa = 0; aa < 4; aa++) {
                            float w = tx.w[aa] * yz;
                            rx += qx[aa] * w;
                            ry += qy[aa] * w;
                            rz += qz[aa] * w;
                        }
                    }
                }
            } else {
                for (int c = 0; c < 4; c++) {
                    int Z = min(max(tz.pre + c, 0), a.gn[2] - 1);
                    for (int bb = 0; bb < 4; bb++) {
                        int Y = min(max(ty.pre + bb, 0), a.gn[1] - 1);
                        float yz = ty.w[bb] * tz.w[c];
                        for (int aa = 0; aa < 4; aa++) {
                            int Xn = min(max(tx.pre + aa, 0), a.gn[0] - 1);
                            size_t id = ((size_t)Z * a.gn[1] + Y) * a.gn[0] + Xn;
                            float w = tx.w[aa] * yz;
                            rx += cp[id] * w;
                            ry += cp[nn + id] * w;
                            rz += cp[2 * nn + id] * w;
                        }
                    }
                }
            }
            world[0] = rx; world[1] = ry; world[2] = rz;
        }
    }
    if (def) {
        float *d = def + (size_t)b * 3 * N;
        d[i] = world[0]; d[N + i] = world[1]; d[2 * N + i] = world[2];
    }
    float pos[3];
    int prev[3];
    float val = nanf("");
    if (DP) {
        for (int r = 0; r < 3; r++)
            pos[r] = (float)((double)a.fijk[r * 4 + 0] * (double)world[0] + (double)a.fijk[r * 4 + 1] * (double)world[1] +
                             (double)a.fijk[r * 4 + 2] * (double)world[2] + (double)a.fijk[r * 4 + 3]);
        prev[0] = (int)floorf(pos[0]); prev[1] = (int)floorf(pos[1]); prev[2] = (int)floorf(pos[2]);
        if (active) {
            double rel[3] = {(double)pos[0] - (double)prev[0], (double)pos[1] - (double)prev[1], (double)pos[2] - (double)prev[2]};
            double bx[2] = {1.0 - rel[0], rel[0]}, by[2] = {1.0 - rel[1], rel[1]}, bz[2] = {1.0 - rel[2], rel[2]};
            if (rel[0] < 0.0) { bx[1] = 0.0; bx[0] = 1.0; }
            if (rel[1] < 0.0) { by[1] = 0.0; by[0] = 1.0; }
            if (rel[2] < 0.0) { bz[1] = 0.0; bz[0] = 1.0; }
            if (prev[0] > -1 && prev[0] + 1 < a.fn[0] && prev[1] > -1 && prev[1] + 1 < a.fn[1] && prev[2] > -1 && prev[2] + 1 < a.fn[2]) {
                double inten = 0.0;
                for (int c = 0; c < 2; c++) {
                    double yt = 0.0;
                    for (int bb = 0; bb < 2; bb++) {
                        const float *p = fl + ((size_t)(prev[2] + c) * a.fn[1] + prev[1] + bb) * a.fn[0] + prev[0];
                        double xt = 0.0;
                        for (int aa = 0; aa < 2; aa++) xt += (double)p[aa] * bx[aa];
                        yt += xt * by[bb];
                    }
                    inten += yt * bz[c];
                }
                val = (float)inten;
            }
        }
    } else {
    // real -> voxel (floating): NiftyReg accumulates in double and rounds to float. fp64 is very slow on this GPU, so the same
    // quantity is computed with float-float arithmetic (correctly rounded to float up to ~2^-48 relative).
    for (int r = 0; r < 3; r++) {
        DF acc = two_prod(a.fijk[r * 4 + 0], world[0]);
        acc = df_add(acc, two_prod(a.fijk[r * 4 + 1], world[1]));
        acc = df_add(acc, two_prod(a.fijk[r * 4 + 2], world[2]));
        acc = df_add(acc, DF{a.fijk[r * 4 + 3], 0.f});
        pos[r] = acc.hi + acc.lo;
    }
    prev[0] = (int)floorf(pos[0]); prev[1] = (int)floorf(pos[1]); prev[2] = (int)floorf(pos[2]);
    if (active) {
        float rel[3] = {pos[0] - (float)prev[0], pos[1] - (float)prev[1], pos[2] - (float)prev[2]};
        for (int k = 0; k < 3; k++) if (rel[k] < 0.f) rel[k] = 0.f;
        DF bx0 = two_sum(1.f, -rel[0]), by0 = two_sum(1.f, -rel[1]), bz0 = two_sum(1.f, -rel[2]);
        DF bx1 = {rel[0], 0.f}, by1 = {rel[1], 0.f}, bz1 = {rel[2], 0.f};
        if (prev[0] > -1 && prev[0] + 1 < a.fn[0] && prev[1] > -1 && prev[1] + 1 < a.fn[1] && prev[2] > -1 &&
            prev[2] + 1 < a.fn[2]) {
            DF inten = {0.f, 0.f};
            for (int c = 0; c < 2; c++) {
                DF yt = {0.f, 0.f};
                for (int bb = 0; bb < 2; bb++) {
                    const float *p = fl + ((size_t)(prev[2] + c) * a.fn[1] + prev[1] + bb) * a.fn[0] + prev[0];
                    DF xt = df_add(df_mul_f(p[0], bx0), df_mul_f(p[1], bx1));
                    yt = df_add(yt, df_mul(xt, bb ? by1 : by0));
                }
                inten = df_add(inten, df_mul(yt, c ? bz1 : bz0));
            }
            val = inten.hi + inten.lo;
        }
    }
    }
    warped[(size_t)b * N + i] = val;

    if (wgrad) {
        float g0 = 0.f, g1 = 0.f, g2 = 0.f;
        if (active && prev[0] >= 0 && prev[0] < a.fn[0] - 1 && prev[1] >= 0 && prev[1] < a.fn[1] - 1 && prev[2] >= 0 &&
            prev[2] < a.fn[2] - 1) {
            float r0 = pos[0] - (float)prev[0], r1 = pos[1] - (float)prev[1], r2 = pos[2] - (float)prev[2];
            float xB[2] = {(float)(1.0 - r0), r0}, yB[2] = {(float)(1.0 - r1), r1}, zB[2] = {(float)(1.0 - r2), r2};
            const float deriv[2] = {-1.f, 1.f};
            for (int c = 0; c < 2; c++) {
                const float *zp = fl + (size_t)(prev[2] + c) * a.fn[0] * a.fn[1];
                float xx = 0.f, yy = 0.f, zz = 0.f;
                for (int bb = 0; bb < 2; bb++) {
                    const float *p = zp + (size_t)(prev[1] + bb) * a.fn[0] + prev[0];
                    float xt = 0.f, yt = 0.f;
                    for (int aa = 0; aa < 2; aa++) {
                        float coeff = p[aa];
                        xt += coeff * deriv[aa];
                        yt += coeff * xB[aa];
                    }
                    xx += xt * yB[bb];
                    yy += yt * deriv[bb];
                    zz += yt * yB[bb];
                }
                g0 += xx * zB[c]; g1 += yy * zB[c]; g2 += zz * deriv[c];
            }
        }
        float *g = wgrad + (size_t)b * 3 * N;
        g[i] = g0; g[N + i] = g1; g[2 * N + i] = g2;
    }
}

// flag the cells of the deformation whose 64 (clamped) control points all map to the same outside half space of the floating image
// (the deformation of every voxel of the cell is a convex combination of these points, the voxel position an affine function of it)
__global__ void cell_flag_k(WarpArgs a, const float *__restrict__ cpp, size_t cppStride, unsigned char *flags, const int *act)
{
    int b = blockIdx.y;
    if (act && !act[b]) return;
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= a.ncell) return;
    int px = c % a.cn[0], py = (c / a.cn[0]) % a.cn[1], pz = c / ((size_t)a.cn[0] * a.cn[1]);
    const float *cp = cpp + (size_t)b * cppStride;
    size_t nn = (size_t)a.gn[0] * a.gn[1] * a.gn[2];
    float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
    for (int cc = 0; cc < 4; cc++) {
        int Z = min(max(pz + cc, 0), a.gn[2] - 1);
        for (int bb = 0; bb < 4; bb++) {
            int Y = min(max(py + bb, 0), a.gn[1] - 1);
            for (int aa = 0; aa < 4; aa++) {
                int X = min(max(px + aa, 0), a.gn[0] - 1);
                size_t id = ((size_t)Z * a.gn[1] + Y) * a.gn[0] + X;
                float wx = cp[id], wy = cp[nn + id], wz = cp[2 * nn + id];
                for (int r = 0; r < 3; r++) {
                    float q = a.fijk[r * 4 + 0] * wx + a.fijk[r * 4 + 1] * wy + a.fijk[r * 4 + 2] * wz + a.fijk[r * 4 + 3];
                    mn[r] = fminf(mn[r], q); mx[r] = fmaxf(mx[r], q);
                }
            }
        }
    }
    bool out = false;
    for (int r = 0; r < 3; r++) out = out || (mx[r] < -0.5f) || (mn[r] > (float)(a.fn[r] - 1) + 0.5f);
    flags[(size_t)b * a.ncell + c] = out ? 1 : 0;
}

void warp_prepare(WarpArgs &a, const std::vector<AxisTab> *tabs)
{
    a.bx = 0;
    a.cflag = nullptr;
    a.fast = false;
    {   // look-up-table structure on every axis?
        bool ok = true;
        for (int k = 0; k < 3 && ok; k++) {
            if (tabs[k].size() < 5) { ok = false; break; }
            for (int l = 0; l < 5; l++) for (int j = 0; j < 4; j++) a.lw[k][l][j] = tabs[k][l].w[j];
            for (size_t i = 0; i < tabs[k].size() && ok; i++) {
                const AxisTab &t = tabs[k][i];
                if (t.pre < 0) continue;
                if (t.pre != (int)(i / 5)) ok = false;
                for (int j = 0; j < 4; j++) if (t.w[j] != a.lw[k][i % 5][j]) ok = false;
            }
        }
        a.lut5 = ok;
    }
    for (int k = 0; k < 3; k++) {
        int mxp = 0;
        for (auto &e : tabs[k]) mxp = std::max(mxp, e.pre);
        a.cn[k] = mxp + 1;
    }
    a.ncell = (size_t)a.cn[0] * a.cn[1] * a.cn[2];
    if (!getenv("F3D_WARP_ROW")) return;   // the shared-memory staged kernel was measured slower than the generic one
    int nx = a.rn[0], ny = a.rn[1];
    int bx = nx >= 128 ? 128 : (nx >= 64 ? 64 : 32);
    int rpb = 128 / bx;
    // footprint check: x nodes per block of bx voxels, y nodes per group of rpb rows
    int maxX = 0, maxY = 0;
    for (int x0 = 0; x0 < nx; x0 += bx) {
        int lo = 1 << 30, hi = -1;
        for (int x = x0; x < nx && x < x0 + bx; x++) if (tabs[0][x].pre >= 0) { lo = std::min(lo, tabs[0][x].pre); hi = std::max(hi, tabs[0][x].pre); }
        if (hi >= 0) maxX = std::max(maxX, std::min(hi + 3, a.gn[0] - 1) - lo + 1);
    }
    for (int y0 = 0; y0 < ny; y0 += rpb) {
        int lo = 1 << 30, hi = -1;
        for (int y = y0; y < ny && y < y0 + rpb; y++) if (tabs[1][y].pre >= 0) { lo = std::min(lo, tabs[1][y].pre); hi = std::max(hi, tabs[1][y].pre); }
        if (hi >= 0) maxY = std::max(maxY, std::min(hi + 3, a.gn[1] - 1) - lo + 1);
    }
    if (maxX <= WARP_XN && maxY <= WARP_YS) a.bx = bx;
}


// ------------------------------------------------------------------ fast numerics warp: one 5x5x5 cell per block
// The deformation of the 125 voxels of a cell is a tensor product of three 5x4 basis tables with the 4x4x4 control points: evaluated
// separably in shared memory (x: 240 values, y: 300 values, z: per voxel), ~30 FMA per voxel instead of 64 x 4. Position -> floating
// voxel and trilinear interpolation in plain float. Cells whose 64 nodes all map outside the floating image (flag array cflag from
// cell_flag_k, or the hull test inside the kernel when there is none) are written as NaN / zero gradient without any other work.
#define WF_NC8 8
template <bool GRAD>
__global__ void __launch_bounds__(128)
    warpf_k(WarpArgs a, const float *__restrict__ cpp, size_t cppStride, const float *__restrict__ flo, size_t floStride, float *warped,
            float *wgrad, const int *act, int ncz)
{
    const size_t N = (size_t)a.rn[0] * a.rn[1] * a.rn[2];
    const int zz = blockIdx.z;
    const int b = zz / ncz, cz = zz - b * ncz;
    const int cy = blockIdx.y, cxb = blockIdx.x * WF_NC8;
    if (act && !act[b]) return;
    const int tid = threadIdx.x;
    const size_t nn = (size_t)a.gn[0] * a.gn[1] * a.gn[2];
    const float *cp = cpp + (size_t)b * cppStride;
    const float *fl = flo + (size_t)b * floStride;
    const int ncx = (a.rn[0] + 4) / 5;
    __shared__ float scp[3 * 64];
    __shared__ float su[240];
    __shared__ float sv[300];
    __shared__ float so[(GRAD ? 4 : 1) * 25 * 40];   // output tile: [comp][lz*5+ly][cell*5+lx]
    __shared__ float smn[2][3], smx[2][3];
    __shared__ int sout;
    const int lx = tid % 5, ly = (tid / 5) % 5, lz = tid / 25;
#pragma unroll 1
    for (int c8 = 0; c8 < WF_NC8; c8++) {
        const int cx = cxb + c8;
        float val = nanf("");
        float g0 = 0.f, g1 = 0.f, g2 = 0.f;
        if (cx < ncx) {
            const bool covered = a.tab[0][cx * 5].pre >= 0 && a.tab[1][cy * 5].pre >= 0 && a.tab[2][cz * 5].pre >= 0;
            bool outside = false;
            if (covered && a.cflag) outside = a.cflag[(size_t)b * a.ncell + ((size_t)cz * a.cn[1] + cy) * a.cn[0] + cx] != 0;
            float wx = 0.f, wy = 0.f, wz = 0.f;
            if (covered && !outside) {
                if (cx + 3 < a.gn[0] && cy + 3 < a.gn[1] && cz + 3 < a.gn[2]) {
                    for (int t = tid; t < 192; t += 128) {
                        int comp = t >> 6, r = t & 63;
                        int c = r >> 4, bb = (r >> 2) & 3, aa = r & 3;
                        scp[t] = cp[(size_t)comp * nn + ((size_t)(cz + c) * a.gn[1] + (cy + bb)) * a.gn[0] + (cx + aa)];
                    }
                } else {
                    for (int t = tid; t < 192; t += 128) {
                        int comp = t >> 6, r = t & 63;
                        int c = r >> 4, bb = (r >> 2) & 3, aa = r & 3;
                        int Z = min(cz + c, a.gn[2] - 1), Y = min(cy + bb, a.gn[1] - 1), X = min(cx + aa, a.gn[0] - 1);
                        scp[t] = cp[(size_t)comp * nn + ((size_t)Z * a.gn[1] + Y) * a.gn[0] + X];
                    }
                }
                __syncthreads();
                if (!a.cflag) {   // no precomputed flags: convex hull test of the 64 nodes here
                    if (tid < 64) {
                        float xx = scp[tid], yy = scp[64 + tid], zz2 = scp[128 + tid];
                        float mn[3], mx[3];
#pragma unroll
                        for (int r = 0; r < 3; r++) {
                            float q = a.fijk[r * 4] * xx + a.fijk[r * 4 + 1] * yy + a.fijk[r * 4 + 2] * zz2 + a.fijk[r * 4 + 3];
                            mn[r] = q; mx[r] = q;
                        }
#pragma unroll
                        for (int o = 16; o > 0; o >>= 1)
#pragma unroll
                            for (int r = 0; r < 3; r++) {
                                mn[r] = fminf(mn[r], __shfl_xor_sync(0xffffffffu, mn[r], o));
                                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], o));
                            }
                        if ((tid & 31) == 0) {
#pragma unroll
                            for (int r = 0; r < 3; r++) { smn[tid >> 5][r] = mn[r]; smx[tid >> 5][r] = mx[r]; }
                        }
                    }
                    __syncthreads();
                    if (tid == 0) {
                        bool out = false;
#pragma unroll
                        for (int r = 0; r < 3; r++)
                            out = out || (fmaxf(smx[0][r], smx[1][r]) < -0.5f) || (fminf(smn[0][r], smn[1][r]) > (float)(a.fn[r] - 1) + 0.5f);
                        sout = out ? 1 : 0;
                    }
                    __syncthreads();
                    outside = sout != 0;
                }
                if (!outside) {
                    for (int t = tid; t < 240; t += 128) {   // x pass: u[lx][comp][c][bb]
                        int lxx = t / 48, r = t - lxx * 48;
                        const float *q = scp + r * 4;
                        const float *w = a.lw[0][lxx];
                        su[t] = __fmaf_rn(w[3], q[3], __fmaf_rn(w[2], q[2], __fmaf_rn(w[1], q[1], w[0] * q[0])));
                    }
                    __syncthreads();
                    for (int t = tid; t < 300; t += 128) {   // y pass: v[lx][ly][comp][c]
                        int lxx = t / 60, r = t - lxx * 60;
                        int lyy = r / 12, rr = r - lyy * 12;   // rr = comp*4 + c
                        const float *q = su + lxx * 48 + rr * 4;
                        const float *w = a.lw[1][lyy];
                        sv[t] = __fmaf_rn(w[3], q[3], __fmaf_rn(w[2], q[2], __fmaf_rn(w[1], q[1], w[0] * q[0])));
                    }
                    __syncthreads();
                    if (tid < 125) {
                        const float *q = sv + (lx * 5 + ly) * 12;
                        const float *w = a.lw[2][lz];
                        wx = __fmaf_rn(w[3], q[3], __fmaf_rn(w[2], q[2], __fmaf_rn(w[1], q[1], w[0] * q[0])));
                        wy = __fmaf_rn(w[3], q[7], __fmaf_rn(w[2], q[6], __fmaf_rn(w[1], q[5], w[0] * q[4])));
                        wz = __fmaf_rn(w[3], q[11], __fmaf_rn(w[2], q[10], __fmaf_rn(w[1], q[9], w[0] * q[8])));
                    }
                }
            }
            if (!outside && tid < 125) {
                float p0 = __fmaf_rn(a.fijk[2], wz, __fmaf_rn(a.fijk[1], wy, __fmaf_rn(a.fijk[0], wx, a.fijk[3])));
                float p1 = __fmaf_rn(a.fijk[6], wz, __fmaf_rn(a.fijk[5], wy, __fmaf_rn(a.fijk[4], wx, a.fijk[7])));
                float p2 = __fmaf_rn(a.fijk[10], wz, __fmaf_rn(a.fijk[9], wy, __fmaf_rn(a.fijk[8], wx, a.fijk[11])));
                float f0 = floorf(p0), f1 = floorf(p1), f2 = floorf(p2);
                int q0 = (int)f0, q1 = (int)f1, q2 = (int)f2;
                if (q0 > -1 && q0 + 1 < a.fn[0] && q1 > -1 && q1 + 1 < a.fn[1] && q2 > -1 && q2 + 1 < a.fn[2]) {
                    const float r0 = p0 - f0, r1 = p1 - f1, r2 = p2 - f2;
                    const size_t s1 = a.fn[0], s2 = (size_t)a.fn[0] * a.fn[1];
                    const float *pb = fl + (size_t)q2 * s2 + (size_t)q1 * s1 + q0;
                    float c000 = __ldg(pb), c100 = __ldg(pb + 1), c010 = __ldg(pb + s1), c110 = __ldg(pb + s1 + 1);
                    float c001 = __ldg(pb + s2), c101 = __ldg(pb + s2 + 1), c011 = __ldg(pb + s2 + s1), c111 = __ldg(pb + s2 + s1 + 1);
                    float x00 = __fmaf_rn(r0, c100 - c000, c000), x10 = __fmaf_rn(r0, c110 - c010, c010);
                    float x01 = __fmaf_rn(r0, c101 - c001, c001), x11 = __fmaf_rn(r0, c111 - c011, c011);
                    float y0 = __fmaf_rn(r1, x10 - x00, x00), y1 = __fmaf_rn(r1, x11 - x01, x01);
                    val = __fmaf_rn(r2, y1 - y0, y0);
                    if (GRAD) {
                        float dx0 = (c100 - c000), dx1 = (c110 - c010), dx2 = (c101 - c001), dx3 = (c111 - c011);
                        float gx0 = __fmaf_rn(r1, dx1 - dx0, dx0), gx1 = __fmaf_rn(r1, dx3 - dx2, dx2);
                        g0 = __fmaf_rn(r2, gx1 - gx0, gx0);
                        float dy0 = (c010 - c000), dy1 = (c110 - c100), dy2 = (c011 - c001), dy3 = (c111 - c101);
                        float gy0 = __fmaf_rn(r0, dy1 - dy0, dy0), gy1 = __fmaf_rn(r0, dy3 - dy2, dy2);
                        g1 = __fmaf_rn(r2, gy1 - gy0, gy0);
                        g2 = y1 - y0;
                    }
                }
            }
            if (tid < 125) {
                const int o = (lz * 5 + ly) * 40 + c8 * 5 + lx;
                so[o] = val;
                if (GRAD) { so[1000 + o] = g0; so[2000 + o] = g1; so[3000 + o] = g2; }
            }
            __syncthreads();   // the next cell reuses scp / su / sv
        }
    }
    __syncthreads();
    // coalesced stores of the tile: rows of up to 40 consecutive voxels
    const int xbase = cxb * 5;
    for (int e = tid; e < 25 * 40; e += 128) {
        const int xo = e % 40, r = e / 40;
        const int x = xbase + xo, y = cy * 5 + r % 5, zv = cz * 5 + r / 5;
        if (x < a.rn[0] && y < a.rn[1] && zv < a.rn[2] && (xbase + xo) / 5 < ncx) {
            const size_t i = ((size_t)zv * a.rn[1] + y) * a.rn[0] + x;
            warped[(size_t)b * N + i] = so[e];
            if (GRAD) {
                float *g = wgrad + (size_t)b * 3 * N;
                g[i] = so[1000 + e]; g[N + i] = so[2000 + e]; g[2 * N + i] = so[3000 + e];
            }
        }
    }
}

void warp(const WarpArgs &a, int B, const float *cpp, size_t cppStride, const float *flo, size_t floStride,
          float *warped, float *wgrad, float *def, const int *act, cudaStream_t st)
{
    if (a.fast && !a.mask && !def && !getenv("F3D_NOFASTWARP")) {
        if (a.lut5) {
            int ncx = (a.rn[0] + 4) / 5, ncy = (a.rn[1] + 4) / 5, ncz = (a.rn[2] + 4) / 5;
            if (a.cflag) cell_flag_k<<<dim3(cdiv(a.ncell, 128), B), 128, 0, st>>>(a, cpp, cppStride, a.cflag, act);
            dim3 grid(cdiv(ncx, WF_NC8), ncy, ncz * B);
            if (wgrad) warpf_k<true><<<grid, 128, 0, st>>>(a, cpp, cppStride, flo, floStride, warped, wgrad, act, ncz);
            else warpf_k<false><<<grid, 128, 0, st>>>(a, cpp, cppStride, flo, floStride, warped, wgrad, act, ncz);
            return;
        }
    }
    size_t N = (size_t)a.rn[0] * a.rn[1] * a.rn[2];
    bool dp = getenv("F3D_WARP_DP") != nullptr;
    if (a.cflag && a.bx == 0 && !def) {
        cell_flag_k<<<dim3(cdiv(a.ncell, 128), B), 128, 0, st>>>(a, cpp, cppStride, a.cflag, act);
    } else if (a.cflag) {
        WarpArgs a2 = a; a2.cflag = nullptr;
        return warp(a2, B, cpp, cppStride, flo, floStride, warped, wgrad, def, act, st);
    }
    if (a.bx > 0) {
        int bx = a.bx, nxb = (a.rn[0] + bx - 1) / bx, rpb = 128 / bx, nyg = (a.rn[1] + rpb - 1) / rpb;
        dim3 grid((unsigned)((size_t)nxb * nyg * a.rn[2]), B);
        if (dp) warp_k<true, true><<<grid, 128, 0, st>>>(a, cpp, cppStride, flo, floStride, warped, wgrad, def, act);
        else warp_k<false, true><<<grid, 128, 0, st>>>(a, cpp, cppStride, flo, floStride, warped, wgrad, def, act);
        return;
    }
    dim3 grid(cdiv(N, 128), B);
    if (dp) warp_k<true, false><<<grid, 128, 0, st>>>(a, cpp, cppStride, flo, floStride, warped, wgrad, def, act);
    else warp_k<false, false><<<grid, 128, 0, st>>>(a, cpp, cppStride, flo, floStride, warped, wgrad, def, act);
}

}  // namespace f3d
