// SPDX-License-Identifier: BSD-3-Clause
// Stage 5 test: BE / LE / Jacobian penalty values, gradients and folding correction vs NiftyReg
#include "test_common.h"
template<class DTYPE> void reg_cubic_spline_jacobian3D(nifti_image *, nifti_image *, mat33 *, DTYPE *, bool, bool);
static nifti_image *copy_img(nifti_image *g)
{
    nifti_image *c = nifti_copy_nim_info(g);
    c->data = malloc(g->nvox * 4);
    memcpy(c->data, g->data, g->nvox * 4);
    return c;
}
int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: test_stage5 target template affine [amp]\n"); return 1; }
    cudaStream_t st = 0;
    nifti_image *ref = load(argv[1]);
    mat44 aff; reg_tool_ReadAffineFile(&aff, argv[3]);
    ImgGeom rg = geom_of(ref);
    nifti_image *rp[3];
    reg_createImagePyramid<float>(ref, rp, 4, 3);
    std::vector<float *> lr; std::vector<ImgGeom> gr;
    { float *d = h2d((float *)ref->data, ref->nvox); build_pyramid(d, rg, 4, 3, lr, gr, st); }
    float amp = argc > 4 ? atof(argv[4]) : 4.f;
    std::mt19937 rng(5);
    const float W_BE = 0.005f / 16.f, W_LE = 0.01f / 3.f, W_JL = 0.0001f;
    for (int l = 0; l < 3; l++) {
        printf("== level %d (amp %g)\n", l, amp);
        float sp[3]; level_spacing(rg, l, sp);
        nifti_image *grid = nullptr;
        reg_createControlPointGrid<float>(&grid, rp[l], sp);
        reg_affine_getDeformationField(&aff, grid);
        std::normal_distribution<float> nd(0.f, amp);
        float *gp = (float *)grid->data;
        for (size_t i = 0; i < grid->nvox; i++) gp[i] += nd(rng);
        GridGeom gg = create_grid(gr[l], sp);
        size_t nn = gg.nnodes();
        GridArgs ga = make_grid_args(gg, gr[l]);
        // GPU buffers
        size_t scr = penalty_scratch_floats(gg);
        float *dcur = h2d(gp, 3 * nn), *dbest = h2d(gp, 3 * nn), *dgrad = dalloc_zero<float>(3 * nn);
        float *dscr = dalloc<float>(scr);
        PenaltyOut *dout = dalloc<PenaltyOut>(1);
        // ---- values on the CPU (type-1 logic of ComputeJacobianBasedPenaltyTerm)
        nifti_image *cg = copy_img(grid);
        double cj = reg_spline_getJacobianPenaltyTerm(cg, rp[l], true);
        int it = 0;
        while (cj != cj && it < 20) { cj = reg_spline_correctFolding(cg, rp[l], true); it++; }
        bool failed = (cj != cj);
        double cbe = reg_spline_approxBendingEnergy(cg);
        double cle = reg_spline_approxLinearEnergy(cg);
        PenaltyBufs pbuf; penalty_bufs_init(pbuf, gg, 1);
        penalty_value(ga, pbuf, dcur, dbest, dgrad, 3 * nn, dscr, scr, nullptr, W_JL, W_BE, W_LE, dout, 1, nullptr, st);
        CK(cudaStreamSynchronize(st));
        PenaltyOut o = d2h(dout, 1)[0];
        auto hc = d2h(dcur, 3 * nn);
        printf("  folding: cpu corrections=%d failed=%d | gpu restored=%d\n", it, (int)failed, o.restored);
        printf("  JAC  cpu %.10g gpu %.10g\n", cj * W_JL, o.jac);
        printf("  BE   cpu %.10g gpu %.10g (rel %.2g)\n", cbe * W_BE, o.be, fabs(cbe * W_BE - o.be) / (fabs(cbe * W_BE) + 1e-30));
        printf("  LE   cpu %.10g gpu %.10g (rel %.2g)\n", cle * W_LE, o.le, fabs(cle * W_LE - o.le) / (fabs(cle * W_LE) + 1e-30));
        if (!failed) cmp("grid after correction", hc.data(), (float *)cg->data, 3 * nn);
        // ---- voxel-level Jacobian penalty + correction (CorrectTransformation, type 2)
        {
            nifti_image *cg2 = copy_img(grid);
            double cv = reg_spline_getJacobianPenaltyTerm(cg2, rp[l], false);
            VoxJacCtx vc; voxjac_init(vc, ga, gg, gr[l]);
            float *dcp = h2d(gp, 3 * nn);
            double gv = voxjac_penalty(vc, dcp, st);
            printf("  voxel jac penalty: cpu %.12g gpu %.12g\n", cv, gv);
            int it2 = 0; double v2 = cv; bool first = true; double gv2 = gv;
            while (v2 != v2 && it2 < 20) { v2 = reg_spline_correctFolding(cg2, rp[l], false); it2++; }
            it2 = 0; first = true;
            while (gv2 != gv2 && it2 < 20) {
                if (!first) gv2 = voxjac_penalty(vc, dcp, st);
                first = false;
                if (gv2 == gv2) { it2++; break; }
                voxjac_correct(vc, dcp, st);
                gv2 = nan("");
                it2++;
            }
            printf("  voxel correction: cpu final %.12g (nan=%d) gpu passes=%d final nan=%d\n", v2, (int)(v2 != v2), it2, (int)(gv2 != gv2));
            if (!(v2 != v2)) { auto hh = d2h(dcp, 3 * nn); cmp("grid after voxel correction", hh.data(), (float *)cg2->data, 3 * nn); }
            voxjac_free(vc);
        }
        // ---- gradients (each separately, on the original grid)
        {
            nifti_image *g1 = copy_img(grid), *gr1 = nifti_copy_nim_info(grid); gr1->data = calloc(grid->nvox, 4);
            reg_spline_approxBendingEnergyGradient(g1, gr1, W_BE);
            float *dc = h2d(gp, 3 * nn), *dg = dalloc_zero<float>(3 * nn);
            be_gradient(ga, dc, dg, 3 * nn, dscr, scr, W_BE, 1, nullptr, st);
            CK(cudaStreamSynchronize(st));
            auto h = d2h(dg, 3 * nn); cmp("BE gradient", h.data(), (float *)gr1->data, 3 * nn);
            auto hcc = d2h(dc, 3 * nn); cmp("grid after BE roundtrip", hcc.data(), (float *)g1->data, 3 * nn);
        }
        {
            nifti_image *gr1 = nifti_copy_nim_info(grid); gr1->data = calloc(grid->nvox, 4);
            reg_spline_approxLinearEnergyGradient(grid, gr1, W_LE);
            float *dc = h2d(gp, 3 * nn), *dg = dalloc_zero<float>(3 * nn);
            le_gradient(ga, dc, dg, 3 * nn, dscr, scr, W_LE, 1, nullptr, st);
            CK(cudaStreamSynchronize(st));
            auto h = d2h(dg, 3 * nn); cmp("LE gradient", h.data(), (float *)gr1->data, 3 * nn);
        }
        {
            nifti_image *gr1 = nifti_copy_nim_info(grid); gr1->data = calloc(grid->nvox, 4);
            reg_spline_getJacobianPenaltyTermGradient(grid, rp[l], gr1, W_JL, true);
            float *dc = h2d(gp, 3 * nn), *dg = dalloc_zero<float>(3 * nn);
            jac_gradient(ga, dc, dg, 3 * nn, dscr, scr, W_JL, 1, nullptr, st);
            CK(cudaStreamSynchronize(st));
            auto h = d2h(dg, 3 * nn); cmp("JAC gradient", h.data(), (float *)gr1->data, 3 * nn);
            size_t ndet = (size_t)(gg.n[0] - 2) * (gg.n[1] - 2) * (gg.n[2] - 2);
            auto hd = d2h(dscr, ndet);
            double sd = 0, sf = 0, sf2 = 0;
            for (size_t i = 0; i < ndet; i++) { double l = log((double)hd[i]); sd += l * l; float lf = logf(hd[i]); sf += (double)lf * lf; sf2 += (double)(lf * lf); }
            double lib = reg_spline_getJacobianPenaltyTerm(grid, rp[l], true);
            {
                std::vector<float> cd(ndet); std::vector<mat33> cm(ndet);
                reg_cubic_spline_jacobian3D<float>(grid, rp[l], cm.data(), cd.data(), true, false);
                size_t nd1 = 0; double mxd = 0;
                for (size_t i = 0; i < ndet; i++) { if (cd[i] != hd[i]) nd1++; mxd = fmax(mxd, fabs((double)cd[i] - hd[i])); }
                printf("  dets: %zu of %zu differ, max diff %.3g (det0 cpu %.9g gpu %.9g)\n", nd1, ndet, mxd, cd[0], hd[0]);
                double s1 = 0; for (size_t i = 0; i < ndet; i++) { double lg = log((double)cd[i]); s1 += lg * lg; }
                printf("  cpu dets -> double-log value %.12g (lib %.12g)\n", s1 / ndet, lib);
            }
            printf("  jac value: lib %.12g | double-log %.12g | float-log %.12g | float-log float-square %.12g\n", lib, sd / ndet, sf / ndet, sf2 / ndet);
        }
    }
    return 0;
}
