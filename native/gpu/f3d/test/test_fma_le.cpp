// SPDX-License-Identifier: BSD-3-Clause
// brute force FMA pattern search for the LE matrix accumulation
#include "test_common.h"
int main(int argc, char **argv)
{
    cudaStream_t st = 0;
    nifti_image *ref = load(argv[1]);
    mat44 aff; reg_tool_ReadAffineFile(&aff, argv[3]);
    ImgGeom rg = geom_of(ref);
    nifti_image *rp[3]; reg_createImagePyramid<float>(ref, rp, 4, 3);
    std::vector<float *> lr; std::vector<ImgGeom> gr;
    { float *d = h2d((float *)ref->data, ref->nvox); build_pyramid(d, rg, 4, 3, lr, gr, st); }
    float amp = atof(argv[4]);
    int l = 1;
    float sp[3]; level_spacing(rg, l, sp);
    nifti_image *grid = nullptr; reg_createControlPointGrid<float>(&grid, rp[l], sp); reg_affine_getDeformationField(&aff, grid);
    std::mt19937 rng(5); std::normal_distribution<float> nd(0.f, amp);
    float *gp = (float *)grid->data; for (size_t i = 0; i < grid->nvox; i++) gp[i] += nd(rng);
    GridGeom gg = create_grid(gr[l], sp); size_t nn = gg.nnodes(); GridArgs ga = make_grid_args(gg, gr[l]);
    size_t scr = penalty_scratch_floats(gg); float *dscr = dalloc<float>(scr);
    const float W = 0.01f / 9.f;
    nifti_image *gr1 = nifti_copy_nim_info(grid); gr1->data = calloc(grid->nvox, 4);
    reg_spline_approxLinearEnergyGradient(grid, gr1, W);
    float *dc = h2d(gp, 3 * nn), *dg = dalloc<float>(3 * nn);
    double best = 1e30; int bestm = -1;
    for (int m = 0; m < 512; m++) {
        set_fuse_masks(m, 216096, 0);
        CK(cudaMemset(dg, 0, 3 * nn * 4));
        le_gradient(ga, dc, dg, 3 * nn, dscr, scr, W, 1, nullptr, st);
        CK(cudaStreamSynchronize(st));
        auto h = d2h(dg, 3 * nn);
        double mx = 0; size_t nd2 = 0;
        for (size_t i = 0; i < 3 * nn; i++) { double d = fabs((double)h[i] - ((float *)gr1->data)[i]); mx = d > mx ? d : mx; nd2 += (d > 0); }
        if (mx < best) { best = mx; bestm = m; printf("mask %d: maxabs %.3g ndiff %zu\n", m, mx, nd2); }
    }
    printf("best mask %d maxabs %.3g\n", bestm, best);
    return 0;
}
