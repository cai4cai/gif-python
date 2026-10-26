// SPDX-License-Identifier: BSD-3-Clause
// micro benchmark of the LNCC value / gradient kernels at pyramid level L2 (B templates), min over repetitions
#include "test_common.h"
#include <chrono>
int main(int argc, char **argv)
{
    cudaStream_t st = 0;
    nifti_image *ref = load(argv[1]), *flo = load(argv[2]);
    mat44 aff; reg_tool_ReadAffineFile(&aff, argv[3]);
    int B = atoi(argv[4]); int lvl = argc > 5 ? atoi(argv[5]) : 2;
    ImgGeom rg = geom_of(ref), fg = geom_of(flo);
    nifti_image *rp[3]; reg_createImagePyramid<float>(ref, rp, 4, 3);
    std::vector<float *> lr, lf; std::vector<ImgGeom> gr, gf;
    { float *d = h2d((float *)ref->data, ref->nvox); build_pyramid(d, rg, 4, 3, lr, gr, st); }
    { float *d = h2d((float *)flo->data, flo->nvox); build_pyramid(d, fg, 4, 3, lf, gf, st); }
    int l = lvl;
    rescale01(lr[l], gr[l].nvox(), st); rescale01(lf[l], gf[l].nvox(), st);
    float sp[3]; level_spacing(rg, l, sp);
    nifti_image *grid = nullptr; reg_createControlPointGrid<float>(&grid, rp[l], sp); reg_affine_getDeformationField(&aff, grid);
    std::mt19937 rng(1); std::normal_distribution<float> nd(0.f, 2.f);
    float *gp = (float *)grid->data; for (size_t i = 0; i < grid->nvox; i++) gp[i] += nd(rng);
    GridGeom gg = create_grid(gr[l], sp); size_t N = gr[l].nvox();
    WarpArgs wa; memset(&wa, 0, sizeof wa);
    for (int i = 0; i < 3; i++) { wa.rn[i] = gr[l].n[i]; wa.gn[i] = gg.n[i]; wa.fn[i] = gf[l].n[i]; }
    for (int r = 0; r < 3; r++) for (int c = 0; c < 4; c++) wa.fijk[r * 4 + c] = gf[l].ijk.m[r][c];
    float gvs[3] = {gg.d[0] / gr[l].d[0], gg.d[1] / gr[l].d[1], gg.d[2] / gr[l].d[2]};
    bool lut = gvs[0] == 5.f;
    std::vector<AxisTab> ht[3]; for (int a = 0; a < 3; a++) { ht[a] = lut ? build_tab_lut(gr[l].n[a], gg.n[a]) : build_tab_generic(gr[l].n[a], gvs[a]); wa.tab[a] = h2d(ht[a].data(), ht[a].size()); }
    warp_prepare(wa, ht);
    size_t nn = gg.nnodes();
    std::vector<float> cps(3 * nn * B); for (int b = 0; b < B; b++) memcpy(&cps[b * 3 * nn], gp, 3 * nn * 4);
    float *dcpp = h2d(cps.data(), cps.size());
    float *dwar = dalloc<float>(N * B), *dgr = dalloc<float>(3 * N * B), *dvg = dalloc<float>(3 * N * B);
    float *dflo = dalloc<float>(gf[l].nvox() * B); for (int b = 0; b < B; b++) CK(cudaMemcpy(dflo + b * gf[l].nvox(), lf[l], gf[l].nvox() * 4, cudaMemcpyDeviceToDevice));
    LnccCtx lc; lncc_init(lc, gr[l].n, gr[l].d, 5.f, lr[l], nullptr, B, getenv("F3D_NOEXACT") == nullptr);
    double *dval = dalloc<double>(B);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto timeit = [&](const char *name, auto fn) {
        float best = 1e30f;
        for (int r = 0; r < 12; r++) { cudaEventRecord(e0, st); fn(); cudaEventRecord(e1, st); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); if (r > 1 && ms < best) best = ms; }
        printf("  %-28s %8.3f ms (B=%d, N=%zu, level %d, exact=%d fused=%d)\n", name, best, B, N, l, (int)lc.exact, (int)lc.fused);
    };
    if (getenv("F3D_WARPFAST")) {   // compare the brick warp with the exact warp
        WarpArgs wf = wa; wf.fast = true;
        float *dw0 = dalloc<float>(N * B), *dg0 = dalloc<float>(3 * N * B);
        warp(wa, B, dcpp, 3 * nn, dflo, gf[l].nvox(), dw0, dg0, nullptr, nullptr, st);
        warp(wf, B, dcpp, 3 * nn, dflo, gf[l].nvox(), dwar, dgr, nullptr, nullptr, st);
        CK(cudaDeviceSynchronize());
        auto a0 = d2h(dw0, N * B), a1 = d2h(dwar, N * B), g0 = d2h(dg0, 3 * N * B), g1 = d2h(dgr, 3 * N * B);
        size_t nanmis = 0, nval = 0; double mx = 0, sum = 0;
        for (size_t i = 0; i < N * B; i++) {
            bool n0 = a0[i] != a0[i], n1 = a1[i] != a1[i];
            if (n0 != n1) nanmis++;
            else if (!n0) { double d = fabs((double)a0[i] - a1[i]); mx = d > mx ? d : mx; sum += d; nval++; }
        }
        double gm = 0; for (size_t i = 0; i < 3 * N * B; i++) { double d = fabs((double)g0[i] - g1[i]); gm = d > gm ? d : gm; }
        printf("  brick warp vs exact: NaN-pattern mismatches %zu, valid %zu, max |dv| %.3g, mean |dv| %.3g, max |dgrad| %.3g\n", nanmis, nval, mx, sum / (nval ? nval : 1), gm);
        wa.fast = true;
    }
    if (getenv("F3D_WARPFAST")) wa.fast = true;
    timeit(wa.fast ? "warp (value) fast" : "warp (value) exact", [&]() { warp(wa, B, dcpp, 3 * nn, dflo, gf[l].nvox(), dwar, nullptr, nullptr, nullptr, st); });
    timeit(wa.fast ? "warp (+gradient) fast" : "warp (+gradient) exact", [&]() { warp(wa, B, dcpp, 3 * nn, dflo, gf[l].nvox(), dwar, dgr, nullptr, nullptr, st); });
    for (int rep = 0; rep < 3; rep++) {
        for (int v = 2; v >= (getenv("F3D_NOEXACT") ? 1 : 2); v--) {
            lc.lfver = v;
            char nm[64];
            snprintf(nm, sizeof nm, "lncc value (kernel v%d)", v); timeit(nm, [&]() { lncc_value(lc, dwar, dval, nullptr, st); });
            snprintf(nm, sizeof nm, "lncc gradient (kernel v%d)", v); timeit(nm, [&]() { lncc_gradient(lc, dwar, dgr, dvg, nullptr, st); });
        }
    }
    auto h = d2h(dval, B); printf("  lncc value[0] = %.17g\n", h[0]);
    return 0;
}
