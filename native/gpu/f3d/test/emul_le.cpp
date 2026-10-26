// SPDX-License-Identifier: BSD-3-Clause
// host emulation of reg_spline_approxLinearEnergyGradient3D / Value3D<float> using the real nifti_mat33_* functions
#include "test_common.h"
extern "C" {
#include "nifti1_io.h"
}
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
    int accMode = atoi(argv[6]);  // 0 unfused 1 fused
    int gMode = atoi(argv[7]);
    int lastFused = argc > 8 ? atoi(argv[8]) : 0;    // gradient accumulate: 0 = fmul,fma,fma,fma(ratio); 1 = all unfused; 2 = all fused (a0*b0 first fused)
    const float W = 0.01f / 9.f;
    size_t nn = (size_t)grid->nx * grid->ny * grid->nz; int nx = grid->nx, ny = grid->ny, nz = grid->nz;
    nifti_image *o1 = nifti_copy_nim_info(grid); o1->data = calloc(grid->nvox, 4);
    reg_spline_approxLinearEnergyGradient(grid, o1, W);
    float basisX[27], basisY[27], basisZ[27]; set_first_order_basis_values(basisX, basisY, basisZ);
    mat33 reor = reg_mat44_to_mat33(&grid->sto_ijk), inv = nifti_mat33_inverse(reor);
    std::vector<float> out(3 * nn, 0.f);
    float ratio = W / (float)nn;
    for (int z = 1; z < nz - 1; z++) for (int y = 1; y < ny - 1; y++) for (int x = 1; x < nx - 1; x++) {
        mat33 m; memset(&m, 0, sizeof m);
        int i = 0;
        if (accMode < 2) {
        for (int c = -1; c < 2; c++) for (int b = -1; b < 2; b++) for (int a = -1; a < 2; a++) {
            size_t id = ((size_t)(z + c) * ny + y + b) * nx + x + a;
            float cx = gp[id], cy = gp[nn + id], cz = gp[2 * nn + id];
            auto ac = [&](float &acc, float bs, float cv) { acc = accMode ? Fm(bs, cv, acc) : Um(bs, cv, acc); };
            ac(m.m[0][0], basisX[i], cx); ac(m.m[1][0], basisY[i], cx); ac(m.m[2][0], basisZ[i], cx);
            ac(m.m[0][1], basisX[i], cy); ac(m.m[1][1], basisY[i], cy); ac(m.m[2][1], basisZ[i], cy);
            ac(m.m[0][2], basisX[i], cz); ac(m.m[1][2], basisY[i], cz);
            if (accMode == 0 && lastFused && (i % 9) >= 6) m.m[2][2] = Fm(basisZ[i], cz, m.m[2][2]); else ac(m.m[2][2], basisZ[i], cz);
            ++i;
        }
        } else {
            // per c-slice partial sums (unfused products), then added to the running total
            float tot[9] = {0}; i = 0;
            for (int c = -1; c < 2; c++) {
                float part[9] = {0};
                for (int b = -1; b < 2; b++) for (int a = -1; a < 2; a++) {
                    size_t id = ((size_t)(z + c) * ny + y + b) * nx + x + a;
                    float cx = gp[id], cy = gp[nn + id], cz = gp[2 * nn + id];
                    float bs[3] = {basisX[i], basisY[i], basisZ[i]}, cv[3] = {cx, cy, cz};
                    for (int q = 0; q < 3; q++) for (int r = 0; r < 3; r++) { volatile float pr = bs[r] * cv[q]; part[r * 3 + q] = part[r * 3 + q] + pr; }
                    ++i;
                }
                for (int k = 0; k < 9; k++) tot[k] = tot[k] + part[k];
            }
            for (int r = 0; r < 3; r++) for (int q = 0; q < 3; q++) m.m[r][q] = tot[r * 3 + q];
        }
        m = nifti_mat33_mul(reor, m);
        mat33 R = nifti_mat33_inverse(nifti_mat33_polar(m));
        m = nifti_mat33_mul(R, m);
        --m.m[0][0]; --m.m[1][1]; --m.m[2][2];
        i = 26;
        for (int c = -1; c < 2; c++) for (int b = -1; b < 2; b++) for (int a = -1; a < 2; a++) {
            size_t id = ((size_t)(z + c) * ny + y + b) * nx + x + a;
            float g0 = (float)(-2.0 * m.m[0][0] * basisX[i]), g1 = (float)(-2.0 * m.m[1][1] * basisY[i]), g2 = (float)(-2.0 * m.m[2][2] * basisZ[i]);
            for (int comp = 0; comp < 3; comp++) {
                float t;
                float r0 = inv.m[comp][0], r1 = inv.m[comp][1], r2 = inv.m[comp][2];
                if (gMode >= 10) {
                    int perm = (gMode - 10) % 6, rf = (gMode - 10) / 6;
                    static const int P[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
                    float gg[3] = {g0, g1, g2}, rr[3] = {r0, r1, r2};
                    volatile float p0 = gg[P[perm][0]] * rr[P[perm][0]];
                    t = Fm(gg[P[perm][2]], rr[P[perm][2]], Fm(gg[P[perm][1]], rr[P[perm][1]], p0));
                    if (rf == 0) out[comp * nn + id] = Fm(ratio, t, out[comp * nn + id]); else { volatile float q = ratio * t; out[comp * nn + id] += q; }
                }
                else if (gMode == 0) { volatile float p0 = g0 * r0; t = Fm(g2, r2, Fm(g1, r1, p0)); out[comp * nn + id] = Fm(ratio, t, out[comp * nn + id]); }
                else if (gMode == 1) { volatile float p0 = r0 * g0, p1 = r1 * g1, p2 = r2 * g2; t = p0 + p1; t = t + p2; volatile float q = ratio * t; out[comp * nn + id] += q; }
                else { volatile float p2 = r1 * g1; t = Fm(r0, g0, p2); t = Fm(r2, g2, t); out[comp * nn + id] = Fm(ratio, t, out[comp * nn + id]); }
            }
            --i;
        }
    }
    size_t nd2 = 0; double mx = 0;
    for (size_t k = 0; k < 3 * nn; k++) { double d = fabs((double)out[k] - ((float *)o1->data)[k]); nd2 += d > 0; mx = d > mx ? d : mx; }
    printf("LE gradient emulation vs library: ndiff %zu of %zu maxabs %g\n", nd2, 3 * nn, mx);
    return 0;
}
