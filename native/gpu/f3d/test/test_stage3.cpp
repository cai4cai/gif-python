// SPDX-License-Identifier: BSD-3-Clause
// Stage 3/4a test: LNCC value and voxel based gradient vs NiftyReg reg_lncc
#include "test_common.h"
int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: test_stage3 target template affine\n"); return 1; }
    cudaStream_t st = 0;
    nifti_image *ref = load(argv[1]), *flo = load(argv[2]);
    mat44 aff; reg_tool_ReadAffineFile(&aff, argv[3]);
    ImgGeom rg = geom_of(ref), fg = geom_of(flo);
    nifti_image *rp[3], *fp[3];
    reg_createImagePyramid<float>(ref, rp, 4, 3);
    reg_createImagePyramid<float>(flo, fp, 4, 3);
    std::vector<float *> lr, lf; std::vector<ImgGeom> gr, gf;
    { float *d = h2d((float *)ref->data, ref->nvox); build_pyramid(d, rg, 4, 3, lr, gr, st); }
    { float *d = h2d((float *)flo->data, flo->nvox); build_pyramid(d, fg, 4, 3, lf, gf, st); }
    std::mt19937 rng(11);
    float amp = argc > 4 ? atof(argv[4]) : 4.f;
    for (int l = 0; l < 3; l++) {
        printf("== level %d\n", l);
        float sp[3]; level_spacing(rg, l, sp);
        nifti_image *grid = nullptr;
        reg_createControlPointGrid<float>(&grid, rp[l], sp);
        reg_affine_getDeformationField(&aff, grid);
        std::normal_distribution<float> nd(0.f, amp);
        float *gp = (float *)grid->data;
        for (size_t i = 0; i < grid->nvox; i++) gp[i] += nd(rng);
        GridGeom gg = create_grid(gr[l], sp);
        size_t N = gr[l].nvox();
        // ---------- CPU
        nifti_image *def = make_def(rp[l]);
        int *mask = (int *)calloc(N, sizeof(int));
        if (getenv("TEST_MASK")) for (size_t i = 0; i < N; i++) { int x = i % gr[l].n[0]; int z = i / ((size_t)gr[l].n[0] * gr[l].n[1]); if (x < gr[l].n[0] / 4 || z > gr[l].n[2] * 3 / 4) mask[i] = -1; }
        nifti_image *war = make_vol(rp[l]);
        nifti_image *wgr = nifti_copy_nim_info(def); wgr->data = calloc(def->nvox, 4);
        nifti_image *vgr = nifti_copy_nim_info(def); vgr->data = calloc(def->nvox, 4);
        reg_lncc lncc;
        lncc.SetKernelStandardDeviation(0, 5.f);
        lncc.SetTimepointWeight(0, 1.0);
        lncc.InitialiseMeasure(rp[l], fp[l], mask, war, wgr, vgr, NULL);   // rescales images to [0,1]
        reg_spline_getDeformationField(grid, def, mask, false, true);
        reg_resampleImage(fp[l], war, def, mask, 1, std::numeric_limits<float>::quiet_NaN());
        double cpuv = lncc.GetSimilarityMeasureValue();
        reg_getImageGradient(fp[l], wgr, def, mask, 1, std::numeric_limits<float>::quiet_NaN(), 0);
        lncc.GetVoxelBasedSimilarityMeasureGradient(0);
        // ---------- GPU
        rescale01(lr[l], N, st); rescale01(lf[l], gf[l].nvox(), st);
        WarpArgs wa; memset(&wa, 0, sizeof wa);
        for (int i = 0; i < 3; i++) { wa.rn[i] = gr[l].n[i]; wa.gn[i] = gg.n[i]; wa.fn[i] = gf[l].n[i]; }
        for (int r = 0; r < 3; r++) for (int c = 0; c < 4; c++) wa.fijk[r * 4 + c] = gf[l].ijk.m[r][c];
        float gvs[3] = {gg.d[0] / gr[l].d[0], gg.d[1] / gr[l].d[1], gg.d[2] / gr[l].d[2]};
        bool lut = gvs[0] == 5.f;
        std::vector<AxisTab> t[3]; AxisTab *dt[3];
        for (int a = 0; a < 3; a++) {
            t[a] = lut ? build_tab_lut(gr[l].n[a], gg.n[a]) : build_tab_generic(gr[l].n[a], gvs[a]);
            dt[a] = h2d(t[a].data(), t[a].size()); wa.tab[a] = dt[a];
        }
        float *dcpp = h2d(gp, grid->nvox);
        float *dwar = dalloc<float>(N), *dgr = dalloc<float>(3 * N), *dvg = dalloc<float>(3 * N);
        int *dmask = getenv("TEST_MASK") ? h2d(mask, N) : nullptr;
        wa.mask = dmask;
        LnccCtx lc; lncc_init(lc, gr[l].n, gr[l].d, 5.f, lr[l], dmask, 1);
        double *dval = dalloc<double>(1);
        warp(wa, 1, dcpp, 0, lf[l], 0, dwar, dgr, nullptr, nullptr, st);
        lncc_value(lc, dwar, dval, nullptr, st);
        lncc_gradient(lc, dwar, dgr, dvg, nullptr, st);
        CK(cudaStreamSynchronize(st));
        double gpuv = d2h(dval, 1)[0];
        printf("  LNCC value  cpu %.9f  gpu %.9f  diff %.3g\n", cpuv, gpuv, gpuv - cpuv);
        auto hw = d2h(dwar, N);
        cmp("warped (rescaled)", hw.data(), (float *)war->data, N);
        auto hv = d2h(dvg, 3 * N);
        cmp("voxel gradient", hv.data(), (float *)vgr->data, 3 * N);
        // ---- stage 4b: spline smoothing of the voxel gradient + node gradient
        {
            float ns[3] = {grid->dx, grid->dy, grid->dz};
            for (int a = 0; a < 3; a++) {
                float cs[3] = {ns[a], ns[a], ns[a]}; bool ax[3] = {false, false, false}; ax[a] = true;
                reg_tools_kernelConvolution(vgr, cs, CUBIC_SPLINE_KERNEL, NULL, NULL, ax);
            }
            nifti_image *node = nifti_copy_nim_info(grid); node->data = calloc(grid->nvox, 4);
            mat44 reor = fp[l]->sto_ijk;
            reg_voxelCentric2NodeCentric(node, vgr, 0.9849f, false, &reor);
            GradCtx gc; gradconv_init(gc, gr[l].n, gr[l].d, ns, 1);
            float *tmp = dalloc<float>(3 * N);
            float *res = gradconv_apply(gc, dvg, tmp, nullptr, st);
            size_t nn = gg.nnodes();
            float *dnode = dalloc<float>(3 * nn);
            vox2node(gr[l], gg, gf[l].ijk, 0.9849f, res, dnode, 1, nullptr, st);
            CK(cudaStreamSynchronize(st));
            auto hs = d2h(res, 3 * N);
            cmp("smoothed voxel gradient", hs.data(), (float *)vgr->data, 3 * N);
            auto hn = d2h(dnode, 3 * nn);
            cmp("node gradient", hn.data(), (float *)node->data, 3 * nn);
        }
    }
    return 0;
}
