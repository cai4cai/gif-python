// SPDX-License-Identifier: BSD-3-Clause
// Host emulation of NiftyReg's BE gradient with explicit fma structure; compared with the library. Used to find the exact
// float operation structure of GCC's vectorised code (the oracle is the library itself).
#include "test_common.h"
#include "f3d_basis_host.h"
#include <cmath>
static inline float F(float a, float b, float c) { return fmaf(a, b, c); }        // fused a*b+c
static inline float U(float a, float b, float c) { volatile float p = a * b; return c + p; }   // unfused
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
    const float W = 0.005f / 256.f;
    int sparse = argc > 9 ? atoi(argv[9]) : 0;
    if (sparse) {
        mat44 mm = grid->sto_xyz; int nx0 = grid->nx, ny0 = grid->ny, nz0 = grid->nz; size_t nn0 = (size_t)nx0 * ny0 * nz0;
        for (int z = 0; z < nz0; z++) for (int y = 0; y < ny0; y++) for (int x = 0; x < nx0; x++) {
            size_t id = ((size_t)z * ny0 + y) * nx0 + x;
            for (int r = 0; r < 3; r++) { float t = (float)y * mm.m[r][1]; t = fmaf((float)x, mm.m[r][0], t); t = fmaf(mm.m[r][2], (float)z, t); gp[r * nn0 + id] = t + mm.m[r][3]; }
        }
        // sparse perturbations
        std::mt19937 r2(9); std::normal_distribution<float> n2(0.f, 5.f);
        for (int k = 0; k < sparse; k++) { int x = 2 + r2() % (nx0 - 4), y = 2 + r2() % (ny0 - 4), z = 2 + r2() % (nz0 - 4); size_t id = ((size_t)z * ny0 + y) * nx0 + x; for (int r = 0; r < 3; r++) gp[r * nn0 + id] += n2(r2); }
    }
    int nx = grid->nx, ny = grid->ny, nz = grid->nz; size_t nn = (size_t)nx * ny * nz;
    nifti_image *g1 = nifti_copy_nim_info(grid); g1->data = malloc(grid->nvox * 4); memcpy(g1->data, grid->data, grid->nvox * 4);
    nifti_image *o1 = nifti_copy_nim_info(grid); o1->data = calloc(grid->nvox, 4);
    reg_spline_approxBendingEnergyGradient(g1, o1, W);
    // ---- emulation
    std::vector<float> disp(3 * nn);
    mat44 m = grid->sto_xyz;
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        size_t id = ((size_t)z * ny + y) * nx + x;
        float init[3];
        for (int r = 0; r < 3; r++) {
            float t = (float)y * m.m[r][1];
            t = fmaf((float)x, m.m[r][0], t);
            t = fmaf(m.m[r][2], (float)z, t);
            init[r] = t + m.m[r][3];
        }
        for (int r = 0; r < 3; r++) disp[r * nn + id] = gp[r * nn + id] - init[r];
    }
    int dmaskInterior = atoi(argv[6]), dmaskBorder = atoi(argv[7]);
    int gatherMode = atoi(argv[8]);     // 0 all fused, 1 all unfused
    std::vector<float> D(18 * nn);
    const float *bs[6] = {h_basis2XX, h_basis2YY, h_basis2ZZ, h_basis2XY, h_basis2YZ, h_basis2XZ};
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        float acc[18] = {0};
        bool interior = x > 0 && y > 0 && z > 0 && x < nx - 1 && y < ny - 1 && z < nz - 1;
        int k = 0;
        for (int c = -1; c < 2; c++) for (int b = -1; b < 2; b++) for (int a = -1; a < 2; a++) {
            int fm = (a == -1) ? dmaskInterior : dmaskBorder;
            if (x + a >= 0 && y + b >= 0 && z + c >= 0 && x + a < nx && y + b < ny && z + c < nz) {
                size_t id = ((size_t)(z + c) * ny + y + b) * nx + x + a;
                for (int comp = 0; comp < 3; comp++) {
                    float cv = disp[comp * nn + id];
                    for (int t = 0; t < 6; t++) {
                        int ai = comp * 6 + t;    // accumulator order in the mask: comp-major (x: XX YY ZZ XY YZ XZ, y..., z...)
                        acc[ai] = ((fm >> ai) & 1) ? F(bs[t][k], cv, acc[ai]) : U(bs[t][k], cv, acc[ai]);
                    }
                }
            }
            ++k;
        }
        float *o = &D[18 * (((size_t)z * ny + y) * nx + x)];
        // output order: XX_x XX_y XX_z YY_x ... ; acc index comp*6 + t
        for (int t = 0; t < 6; t++) for (int comp = 0; comp < 3; comp++) {
            float v = acc[comp * 6 + t];
            o[t * 3 + comp] = (t >= 3) ? (float)(2.0 * v) : v;
        }
    }
    float ratio = W / (float)nn;
    std::vector<float> out(3 * nn, 0.f);
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        float gv[3] = {0, 0, 0};
        int a = 0;
        for (int Z = z - 1; Z < z + 2; Z++) for (int Y = y - 1; Y < y + 2; Y++) for (int X = x - 1; X < x + 2; X++) {
            if (X >= 0 && Y >= 0 && Z >= 0 && X < nx && Y < ny && Z < nz) {
                const float *p = &D[18 * (((size_t)Z * ny + Y) * nx + X)];
                for (int t = 0; t < 6; t++) for (int comp = 0; comp < 3; comp++)
                    gv[comp] = gatherMode == 0 ? F(p[t * 3 + comp], bs[t][a], gv[comp]) : U(p[t * 3 + comp], bs[t][a], gv[comp]);
            }
            a++;
        }
        size_t id = ((size_t)z * ny + y) * nx + x;
        for (int comp = 0; comp < 3; comp++) out[comp * nn + id] = F(ratio, gv[comp], 0.f);
    }
    size_t nd2 = 0; double mx = 0;
    for (size_t i = 0; i < 3 * nn; i++) { double d = fabs((double)out[i] - ((float *)o1->data)[i]); nd2 += d > 0; mx = d > mx ? d : mx; }
    printf("emulation vs library: ndiff %zu of %zu maxabs %g\n", nd2, 3 * nn, mx);
    return 0;
}
