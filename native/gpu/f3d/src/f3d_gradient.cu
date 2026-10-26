// SPDX-License-Identifier: BSD-3-Clause
// Voxel based gradient smoothing (cubic spline kernel) and voxel->node conversion (NiftyReg reg_voxelCentric2NodeCentric)
#include "f3d_internal.h"
#include <cmath>
#include <algorithm>

namespace f3d {

void gradconv_init(GradCtx &g, const int dims[3], const float pixdim[3], const float nodeSpacing[3], int B)
{
    for (int i = 0; i < 3; i++) g.dims[i] = dims[i];
    g.N = (size_t)dims[0] * dims[1] * dims[2];
    g.B = B;
    for (int a = 0; a < 3; a++) {
        g.t[a] = make_taps(nodeSpacing[a], pixdim[a], 1);
        if (g.t[a].radius > 0) {
            auto d = density_table(g.t[a], dims[a]);
            g.dens[a] = h2d(d.data(), d.size());
        }
    }
}
void gradconv_free(GradCtx &g)
{
    for (int a = 0; a < 3; a++) { free_taps(g.t[a]); if (g.dens[a]) pool_free(g.dens[a]); g.dens[a] = nullptr; if (g.lst[a]) pool_free(g.lst[a]); g.lst[a] = nullptr; g.lcnt[a] = 0; }
}
void gradconv_prune(GradCtx &g, const ImgGeom &ref, const GridGeom &grid)
{
    if (getenv("F3D_NOPRUNE")) return;
    for (int a = 0; a < 3; a++) if (g.t[a].radius <= 0 || g.dims[a] <= 1) return;   // skipped axes: keep the plain path
    const int nx = g.dims[0], ny = g.dims[1], nz = g.dims[2];
    M44 T = m44_mul(ref.ijk, grid.xyz);
    std::vector<unsigned char> m3((size_t)nx * ny * nz, 0);
    for (int z = 0; z < grid.n[2]; z++)
        for (int y = 0; y < grid.n[1]; y++)
            for (int x = 0; x < grid.n[0]; x++) {
                double nc[3] = {(double)(float)x, (double)(float)y, (double)(float)z};
                int pre[2][3];
                for (int v = 0; v < 2; v++)
                    for (int r = 0; r < 3; r++) {
                        double d;
                        double t0 = (double)T.m[r][0] * nc[0];
                        if (v == 0) d = t0 + (double)T.m[r][1] * nc[1] + (double)T.m[r][2] * nc[2] + (double)T.m[r][3];
                        else { d = fma((double)T.m[r][1], nc[1], t0); d = fma((double)T.m[r][2], nc[2], d); d = d + (double)T.m[r][3]; }
                        float vc = (float)d;
                        pre[v][r] = (vc >= 0) ? (int)vc : (int)floorf(vc);
                    }
                for (int v = 0; v < 2; v++)
                    for (int c = 0; c < 2; c++) for (int b = 0; b < 2; b++) for (int a2 = 0; a2 < 2; a2++) {
                        int X = pre[v][0] + a2, Y = pre[v][1] + b, Z = pre[v][2] + c;
                        if (X > -1 && X < nx && Y > -1 && Y < ny && Z > -1 && Z < nz) m3[((size_t)Z * ny + Y) * nx + X] = 1;
                    }
            }
    // dilations: V2 = dilate along z(V3), V1 = dilate along y(V2)
    auto dilate = [&](const std::vector<unsigned char> &in, int axis, int radius) {
        std::vector<unsigned char> out(in.size(), 0);
        size_t st = axis == 0 ? 1 : (axis == 1 ? (size_t)nx : (size_t)nx * ny);
        int n = g.dims[axis];
        for (size_t i = 0; i < in.size(); i++) {
            if (!in[i]) continue;
            int p = (int)((i / st) % n);
            int lo = std::max(p - radius, 0), hi = std::min(p + radius, n - 1);
            for (int k = lo; k <= hi; k++) out[i + (ptrdiff_t)(k - p) * (ptrdiff_t)st] = 1;
        }
        return out;
    };
    auto m2 = dilate(m3, 2, g.t[2].radius);
    auto m1 = dilate(m2, 1, g.t[1].radius);
    const std::vector<unsigned char> *ms[3] = {&m1, &m2, &m3};   // pass x, y, z outputs
    for (int a = 0; a < 3; a++) {
        std::vector<unsigned> l;
        for (size_t i = 0; i < ms[a]->size(); i++) if ((*ms[a])[i]) l.push_back((unsigned)i);
        g.lcnt[a] = (int)l.size();
        g.lst[a] = h2d(l.data(), std::max<size_t>(l.size(), 1));
    }
}
float *gradconv_apply(GradCtx &g, float *vg, float *tmp, const int *act, cudaStream_t st)
{
    size_t bs = 3 * g.N;
    conv_axis(vg, tmp, 3, g.B, bs, g.dims, 0, g.t[0], st, g.exact, g.dens[0], act, g.lst[0], g.lcnt[0]);
    conv_axis(tmp, vg, 3, g.B, bs, g.dims, 1, g.t[1], st, g.exact, g.dens[1], act, g.lst[1], g.lcnt[1]);
    conv_axis(vg, tmp, 3, g.B, bs, g.dims, 2, g.t[2], st, g.exact, g.dens[2], act, g.lst[2], g.lcnt[2]);
    return tmp;
}

struct V2NArgs {
    int vn[3];
    int gn[3];
    float tr[12];   // node index -> voxel index (3x4)
    float re[9];    // reorientation (3x3)
    float weight;
};
__global__ void vox2node_k(V2NArgs a, const float *vg, float *ng, int B, const int *act)
{
    size_t nn = (size_t)a.gn[0] * a.gn[1] * a.gn[2];
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nn) return;
    int b = blockIdx.y;
    if (act && !act[b]) return;
    size_t N = (size_t)a.vn[0] * a.vn[1] * a.vn[2];
    const float *v = vg + (size_t)b * 3 * N;
    int x = i % a.gn[0], y = (i / a.gn[0]) % a.gn[1], z = i / ((size_t)a.gn[0] * a.gn[1]);
    float nc[3] = {(float)x, (float)y, (float)z}, vc[3];
    for (int r = 0; r < 3; r++)
        vc[r] = (float)((double)a.tr[r * 4 + 0] * (double)nc[0] + (double)a.tr[r * 4 + 1] * (double)nc[1] +
                        (double)a.tr[r * 4 + 2] * (double)nc[2] + (double)a.tr[r * 4 + 3]);
    int pre[3];
    for (int r = 0; r < 3; r++) pre[r] = (vc[r] >= 0) ? (int)vc[r] : (int)floorf(vc[r]);
    float bx[2], by[2], bz[2];
    bx[1] = vc[0] - (float)pre[0]; bx[0] = 1.f - bx[1];
    by[1] = vc[1] - (float)pre[1]; by[0] = 1.f - by[1];
    bz[1] = vc[2] - (float)pre[2]; bz[0] = 1.f - bz[1];
    float iv[3] = {0.f, 0.f, 0.f};
    for (int c = 0; c < 2; c++) {
        int Z = pre[2] + c;
        if (Z > -1 && Z < a.vn[2]) {
            for (int b2 = 0; b2 < 2; b2++) {
                int Y = pre[1] + b2;
                if (Y > -1 && Y < a.vn[1]) {
                    for (int a2 = 0; a2 < 2; a2++) {
                        int X = pre[0] + a2;
                        if (X > -1 && X < a.vn[0]) {
                            size_t id = ((size_t)Z * a.vn[1] + Y) * a.vn[0] + X;
                            float lw = bx[a2] * by[b2];
                            lw *= bz[c];
                            iv[0] += lw * v[id];
                            iv[1] += lw * v[N + id];
                            iv[2] += lw * v[2 * N + id];
                        }
                    }
                }
            }
        }
    }
    float ro[3];
    ro[0] = a.re[0 * 3 + 0] * iv[0] + a.re[1 * 3 + 0] * iv[1] + a.re[2 * 3 + 0] * iv[2];
    ro[1] = a.re[0 * 3 + 1] * iv[0] + a.re[1 * 3 + 1] * iv[1] + a.re[2 * 3 + 1] * iv[2];
    ro[2] = a.re[0 * 3 + 2] * iv[0] + a.re[1 * 3 + 2] * iv[1] + a.re[2 * 3 + 2] * iv[2];
    float *o = ng + (size_t)b * 3 * nn;
    o[i] = ro[0] * a.weight;
    o[nn + i] = ro[1] * a.weight;
    o[2 * nn + i] = ro[2] * a.weight;
}

void vox2node(const ImgGeom &ref, const GridGeom &grid, const M44 &floIJK, float weight, const float *vg, float *nodeGrad,
              int B, const int *act, cudaStream_t st)
{
    V2NArgs a;
    for (int i = 0; i < 3; i++) { a.vn[i] = ref.n[i]; a.gn[i] = grid.n[i]; }
    // transformation = ref.ijk * grid.xyz (reg_mat44_mul, double accumulation)
    M44 T = m44_mul(ref.ijk, grid.xyz);
    for (int r = 0; r < 3; r++) for (int c = 0; c < 4; c++) a.tr[r * 4 + c] = T.m[r][c];
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) a.re[r * 3 + c] = floIJK.m[r][c];
    float w = weight;
    for (int i = 0; i < 3; i++) {
        float ratio = sqrtf(grid.xyz.m[i][0] * grid.xyz.m[i][0] + grid.xyz.m[i][1] * grid.xyz.m[i][1] + grid.xyz.m[i][2] * grid.xyz.m[i][2]);
        ratio /= ref.d[i];
        w *= ratio;
    }
    a.weight = w;
    dim3 g(cdiv(grid.nnodes(), 128), B);
    vox2node_k<<<g, 128, 0, st>>>(a, vg, nodeGrad, B, act);
}

}  // namespace f3d
