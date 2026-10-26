// SPDX-License-Identifier: BSD-3-Clause
// host emulation of reg_spline_approxBendingEnergyValue3D<float> with explicit float operation structure
#include "test_common.h"
#include "f3d_basis_host.h"
#include <cmath>
static inline float Fm(float a, float b, float c) { return fmaf(a, b, c); }
static inline float Um(float a, float b, float c) { volatile float p = a * b; return c + p; }
int main(int argc, char **argv)
{
    nifti_image *ref = load(argv[1]);
    mat44 aff; reg_tool_ReadAffineFile(&aff, argv[3]);
    ImgGeom rg = geom_of(ref);
    nifti_image *rp[3]; reg_createImagePyramid<float>(ref, rp, 4, 3);
    int l = atoi(argv[5]); float sp[3]; level_spacing(rg, l, sp);
    nifti_image *grid = nullptr; reg_createControlPointGrid<float>(&grid, rp[l], sp); reg_affine_getDeformationField(&aff, grid);
    std::mt19937 rng(5); std::normal_distribution<float> nd(0.f, atof(argv[4]));
    float *gp = (float *)grid->data; for (size_t i = 0; i < grid->nvox; i++) gp[i] += nd(rng);
    int accMode = atoi(argv[6]);   // 0 all unfused, 1 all fused
    int sumMode = atoi(argv[7]);   // 0 separate products, 1 fma chain
    double lib = reg_spline_approxBendingEnergy(grid);
    int nx = grid->nx, ny = grid->ny, nz = grid->nz; size_t nn = (size_t)nx * ny * nz;
    const float *bs[6] = {h_basis2XX, h_basis2YY, h_basis2ZZ, h_basis2XY, h_basis2YZ, h_basis2XZ};
    double constraint = 0.0;
    for (int z = 1; z < nz - 1; z++) for (int y = 1; y < ny - 1; y++) for (int x = 1; x < nx - 1; x++) {
        float acc[18] = {0};
        int k = 0;
        for (int c = -1; c < 2; c++) for (int b = -1; b < 2; b++) for (int a = -1; a < 2; a++) {
            size_t id = ((size_t)(z + c) * ny + y + b) * nx + x + a;
            for (int comp = 0; comp < 3; comp++) {
                float cv = gp[comp * nn + id];
                for (int t = 0; t < 6; t++) {
                    int ai = comp * 6 + t;
                    acc[ai] = accMode ? Fm(bs[t][k], cv, acc[ai]) : Um(bs[t][k], cv, acc[ai]);
                }
            }
            k++;
        }
        auto sq = [&](float s0, float v, float v2) { return v; };
        // acc index: comp*6 + t ; t: XX YY ZZ XY YZ XZ
        auto XX = [&](int c) { return acc[c * 6 + 0]; }; auto YY = [&](int c) { return acc[c * 6 + 1]; }; auto ZZ = [&](int c) { return acc[c * 6 + 2]; };
        auto XY = [&](int c) { return acc[c * 6 + 3]; }; auto YZ = [&](int c) { return acc[c * 6 + 4]; }; auto XZ = [&](int c) { return acc[c * 6 + 5]; };
        double v = 0;
        for (int c = 0; c < 3; c++) {
            float xx = XX(c), yy = YY(c), zz = ZZ(c), xy = XY(c), yz = YZ(c), xz = XZ(c);
            float s1, s2;
            if (sumMode == 0) { s1 = xx * xx; volatile float t1 = yy * yy; s1 = s1 + t1; volatile float t2 = zz * zz; s1 = s1 + t2; volatile float u1 = xy * xy, u2 = yz * yz, u3 = xz * xz; s2 = u1 + u2; s2 = s2 + u3; }
            else { volatile float t0 = xx * xx; s1 = Fm(zz, zz, Fm(yy, yy, t0)); volatile float u0 = xy * xy; s2 = Fm(xz, xz, Fm(yz, yz, u0)); }
            if (c == 0) v = (double)s1 + 2.0 * (double)s2;
            else { v += (double)s1; v += 2.0 * (double)s2; }
        }
        constraint += v;
    }
    double emu = constraint / (double)grid->nvox;
    printf("emulation %.17g library %.17g  rel diff %.3g\n", emu, lib, fabs(emu - lib) / lib);
    return 0;
}
