// SPDX-License-Identifier: BSD-3-Clause
// compare the float-float warp with the double precision warp (and thus NiftyReg) on many random grids
#include "test_common.h"
int main(int argc, char **argv)
{
    cudaStream_t st = 0;
    nifti_image *ref = load(argv[1]), *flo = load(argv[2]);
    mat44 aff; reg_tool_ReadAffineFile(&aff, argv[3]);
    ImgGeom rg = geom_of(ref), fg = geom_of(flo);
    nifti_image *rp[3]; reg_createImagePyramid<float>(ref, rp, 4, 3);
    std::vector<float *> lf; std::vector<ImgGeom> gf, gr; std::vector<float *> lr;
    { float *d = h2d((float *)flo->data, flo->nvox); build_pyramid(d, fg, 4, 3, lf, gf, st); }
    { float *d = h2d((float *)ref->data, ref->nvox); build_pyramid(d, rg, 4, 3, lr, gr, st); }
    int nrep = atoi(argv[4]);
    std::mt19937 rng(3);
    size_t total = 0, bad = 0, badnan = 0;
    for (int l = 2; l >= 0; l--) {
        float sp[3]; level_spacing(rg, l, sp);
        GridGeom gg = create_grid(gr[l], sp);
        nifti_image *grid = nullptr; reg_createControlPointGrid<float>(&grid, rp[l], sp); reg_affine_getDeformationField(&aff, grid);
        size_t N = gr[l].nvox();
        WarpArgs wa; memset(&wa, 0, sizeof wa);
        for (int i = 0; i < 3; i++) { wa.rn[i] = gr[l].n[i]; wa.gn[i] = gg.n[i]; wa.fn[i] = gf[l].n[i]; }
        for (int r = 0; r < 3; r++) for (int c = 0; c < 4; c++) wa.fijk[r * 4 + c] = gf[l].ijk.m[r][c];
        float gvs[3] = {gg.d[0] / gr[l].d[0], gg.d[1] / gr[l].d[1], gg.d[2] / gr[l].d[2]};
        bool lut = gvs[0] == 5.f;
        for (int a = 0; a < 3; a++) { auto t = lut ? build_tab_lut(gr[l].n[a], gg.n[a]) : build_tab_generic(gr[l].n[a], gvs[a]); wa.tab[a] = h2d(t.data(), t.size()); }
        float *dw1 = dalloc<float>(N), *dw2 = dalloc<float>(N);
        std::vector<float> base((float *)grid->data, (float *)grid->data + grid->nvox);
        for (int r = 0; r < nrep; r++) {
            std::normal_distribution<float> nd(0.f, 1.f + 6.f * (r % 5));
            std::vector<float> g = base;
            for (auto &v : g) v += nd(rng);
            float *dc = h2d(g.data(), g.size());
            setenv("F3D_WARP_DP", "1", 1); warp(wa, 1, dc, 0, lf[l], 0, dw1, nullptr, nullptr, nullptr, st);
            unsetenv("F3D_WARP_DP"); warp(wa, 1, dc, 0, lf[l], 0, dw2, nullptr, nullptr, nullptr, st);
            CK(cudaStreamSynchronize(st));
            auto h1 = d2h(dw1, N), h2 = d2h(dw2, N);
            for (size_t i = 0; i < N; i++) { total++; if (h1[i] != h2[i] && !(h1[i] != h1[i] && h2[i] != h2[i])) { bad++; if (bad < 4) printf("  diff level %d voxel %zu: dp %.9g df %.9g\n", l, i, h1[i], h2[i]); } }
            cudaFree(dc);
        }
    }
    printf("voxels compared %zu, mismatches %zu\n", total, bad);
    return 0;
}
