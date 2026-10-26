// SPDX-License-Identifier: BSD-3-Clause
// Stage 1/2 test: pyramid, grid init, grid refinement, deformation+resample+gradient vs NiftyReg CPU
#include "f3d_internal.h"
#include "_reg_tools.h"
#include "_reg_localTrans.h"
#include "_reg_resampling.h"
#include "_reg_globalTrans.h"
#include "_reg_ReadWriteMatrix.h"
#include <random>
#include <cstdio>
using namespace f3d;

static nifti_image *load(const char *fn)
{
    nifti_image *im = nifti_image_read(fn, true);
    if (!im) { fprintf(stderr, "cannot read %s\n", fn); exit(1); }
    nifti_free_extensions(im);
    im->scl_inter = 0; im->scl_slope = 1;
    if (im->datatype != NIFTI_TYPE_FLOAT32) reg_tools_changeDatatype<float>(im);
    nifti_update_dims_from_array(im);
    return im;
}
static M44 to_m44(const mat44 &m) { M44 r; for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) r.m[i][j] = m.m[i][j]; return r; }
static ImgGeom geom_of(nifti_image *im)
{
    if (im->sform_code <= 0) { fprintf(stderr, "sform required\n"); exit(1); }
    return make_geom(im->nx, im->ny, im->nz, im->dx, im->dy, im->dz, to_m44(im->sto_xyz));
}
static void cmp(const char *name, const float *a, const float *b, size_t n)
{
    double mx = 0, sum = 0; size_t nbad = 0, nan = 0;
    for (size_t i = 0; i < n; i++) {
        if (a[i] != a[i] || b[i] != b[i]) { if ((a[i] != a[i]) != (b[i] != b[i])) nbad++; nan += (a[i] != a[i]); continue; }
        double d = fabs((double)a[i] - b[i]); mx = d > mx ? d : mx; sum += d;
    }
    printf("  %-28s n=%zu maxabs=%.3g mean=%.3g nanMismatch=%zu (nan=%zu)\n", name, n, mx, sum / n, nbad, nan);
}
static bool geom_eq(const ImgGeom &g, nifti_image *im, const char *nm)
{
    bool ok = g.n[0] == im->nx && g.n[1] == im->ny && g.n[2] == im->nz && g.d[0] == im->dx && g.d[1] == im->dy && g.d[2] == im->dz;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) ok = ok && g.xyz.m[i][j] == im->sto_xyz.m[i][j];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) if (g.ijk.m[i][j] != im->sto_ijk.m[i][j]) { ok = false; }
    printf("  geometry %s: %s (gpu %dx%dx%d d=%g,%g,%g)\n", nm, ok ? "MATCH" : "DIFF", g.n[0], g.n[1], g.n[2], g.d[0], g.d[1], g.d[2]);
    return ok;
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: test_stage1 target template affine\n"); return 1; }
    cudaStream_t st = 0;
    nifti_image *ref = load(argv[1]), *flo = load(argv[2]);
    mat44 aff; reg_tool_ReadAffineFile(&aff, argv[3]);
    ImgGeom rg = geom_of(ref), fg = geom_of(flo);

    // ---- stage 1a: pyramids
    printf("== pyramid\n");
    nifti_image *rp[3], *fp[3];
    reg_createImagePyramid<float>(ref, rp, 4, 3);
    reg_createImagePyramid<float>(flo, fp, 4, 3);
    for (int which = 0; which < 2; which++) {
        nifti_image *im = which ? flo : ref; ImgGeom g = which ? fg : rg; nifti_image **pp = which ? fp : rp;
        float *d = h2d((float *)im->data, im->nvox);
        std::vector<float *> lv; std::vector<ImgGeom> gl;
        build_pyramid(d, g, 4, 3, lv, gl, st);
        for (int l = 0; l < 3; l++) {
            printf(" %s level %d\n", which ? "floating" : "reference", l);
            geom_eq(gl[l], pp[l], "level");
            auto h = d2h(lv[l], gl[l].nvox());
            cmp("image", h.data(), (float *)pp[l]->data, gl[l].nvox());
        }
        // rescale test on level 2
        for (int l = 0; l < 3; l++) {
            reg_intensityRescale(pp[l], 0, 0.f, 1.f);
            rescale01(lv[l], gl[l].nvox(), st);
            auto h = d2h(lv[l], gl[l].nvox());
            cmp("rescaled image", h.data(), (float *)pp[l]->data, gl[l].nvox());
        }
        if (which == 0) {
            // keep for later stages
            // ---- stage 1b: grid creation + refinement
            printf("== grid\n");
            float sp[3];
            sp[0] = 5.f * rg.d[0] * powf(2.f, 3.f); sp[1] = 5.f * rg.d[1] * powf(2.f, 3.f); sp[2] = 5.f * rg.d[2] * powf(2.f, 3.f);
            nifti_image *grid = nullptr;
            reg_createControlPointGrid<float>(&grid, rp[0], sp);
            reg_affine_getDeformationField(&aff, grid);
            GridGeom gg = create_grid(gl[0], sp);
            printf("  grid0 dims cpu %dx%dx%d gpu %dx%dx%d\n", grid->nx, grid->ny, grid->nz, gg.n[0], gg.n[1], gg.n[2]);
            bool ok = true;
            for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) ok = ok && gg.xyz.m[i][j] == grid->sto_xyz.m[i][j];
            printf("  grid0 sto_xyz %s, spacing %g,%g,%g vs %g,%g,%g\n", ok ? "MATCH" : "DIFF", gg.d[0], gg.d[1], gg.d[2], grid->dx, grid->dy, grid->dz);
            M44 A = to_m44(aff);
            auto hg = grid_init_affine(gg, A);
            cmp("grid affine init", hg.data(), (float *)grid->data, hg.size());
            // perturb and refine through the levels
            std::mt19937 rng(1);
            std::normal_distribution<float> nd(0.f, 3.f);
            float *gp = (float *)grid->data;
            for (size_t i = 0; i < grid->nvox; i++) { float v = nd(rng); gp[i] += v; hg[i] += v; }
            float *dold = h2d(hg.data(), hg.size());
            for (int l = 1; l < 3; l++) {
                GridGeom gn = refine_grid_geom(gg, gl[l]);
                reg_spline_refineControlPointGrid(grid, rp[l]);
                float *dnew = dalloc<float>(3 * gn.nnodes());
                grid_refine(dold, gg, dnew, gn, 1, st);
                CK(cudaStreamSynchronize(st));
                printf(" refine to level %d: cpu %dx%dx%d gpu %dx%dx%d\n", l, grid->nx, grid->ny, grid->nz, gn.n[0], gn.n[1], gn.n[2]);
                ok = true;
                for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) ok = ok && gn.xyz.m[i][j] == grid->sto_xyz.m[i][j];
                printf("  sto_xyz %s dx %g vs %g\n", ok ? "MATCH" : "DIFF", gn.d[0], grid->dx);
                auto h = d2h(dnew, 3 * gn.nnodes());
                cmp("refined grid", h.data(), (float *)grid->data, 3 * gn.nnodes());
                cudaFree(dold); dold = dnew; gg = gn;
            }
        }
    }

    // ---- stage 2: deformation + resampling + gradient, per level, with a perturbed grid
    printf("== deformation/resample/gradient\n");
    {
        // flo pyramid un-rescaled for this test (recreate)
        nifti_image *fp2[3], *rp2[3];
        reg_createImagePyramid<float>(flo, fp2, 4, 3);
        reg_createImagePyramid<float>(ref, rp2, 4, 3);
        std::vector<float *> lvf; std::vector<ImgGeom> glf;
        {
            float *d = h2d((float *)flo->data, flo->nvox);
            build_pyramid(d, fg, 4, 3, lvf, glf, st);
        }
        std::mt19937 rng(7);
        for (int l = 0; l < 3; l++) {
            ImgGeom rgl = geom_of(rp2[l]);
            float sp[3];
            float f = powf(2.f, (float)(2 - l));
            sp[0] = 5.f * 0.9375f * f * 2.f; sp[1] = sp[0]; sp[2] = 5.f * rg.d[2] * f * 2.f;
            sp[0] = 5.f * rg.d[0] * 2.f * f; sp[1] = 5.f * rg.d[1] * 2.f * f; sp[2] = 5.f * rg.d[2] * 2.f * f;
            // spacing for z at level 0 is 10 voxels in NiftyReg (z not downsampled) - mimic generic: use the grid derived from level 0 spacing
            nifti_image *grid = nullptr;
            reg_createControlPointGrid<float>(&grid, rp2[l], sp);
            reg_affine_getDeformationField(&aff, grid);
            std::normal_distribution<float> nd(0.f, 4.f);
            float *gp = (float *)grid->data;
            for (size_t i = 0; i < grid->nvox; i++) gp[i] += nd(rng);
            GridGeom gg = create_grid(rgl, sp);
            // CPU
            nifti_image *def = nifti_copy_nim_info(rp2[l]);
            def->dim[0] = def->ndim = 5; def->dim[4] = def->nt = 1; def->dim[5] = def->nu = 3; def->nvox = (size_t)rp2[l]->nx * rp2[l]->ny * rp2[l]->nz * 3;
            def->nbyper = 4; def->datatype = NIFTI_TYPE_FLOAT32; def->data = calloc(def->nvox, 4);
            size_t N = rgl.nvox();
            int *mask = (int *)calloc(N, sizeof(int));
            reg_spline_getDeformationField(grid, def, mask, false, true);
            nifti_image *war = nifti_copy_nim_info(rp2[l]);
            war->datatype = NIFTI_TYPE_FLOAT32; war->nbyper = 4; war->scl_slope = 1; war->scl_inter = 0; war->data = calloc(N, 4);
            reg_resampleImage(fp2[l], war, def, mask, 1, std::numeric_limits<float>::quiet_NaN());
            nifti_image *grad = nifti_copy_nim_info(def);
            grad->data = calloc(def->nvox, 4);
            reg_getImageGradient(fp2[l], grad, def, mask, 1, std::numeric_limits<float>::quiet_NaN(), 0);
            // GPU
            WarpArgs wa; memset(&wa, 0, sizeof wa);
            for (int i = 0; i < 3; i++) { wa.rn[i] = rgl.n[i]; wa.gn[i] = gg.n[i]; wa.fn[i] = glf[l].n[i]; }
            for (int r = 0; r < 3; r++) for (int c = 0; c < 4; c++) wa.fijk[r * 4 + c] = glf[l].ijk.m[r][c];
            float gvs[3] = {gg.d[0] / rgl.d[0], gg.d[1] / rgl.d[1], gg.d[2] / rgl.d[2]};
            bool lut = gvs[0] == 5.f;
            std::vector<AxisTab> t[3]; AxisTab *dt[3];
            for (int a = 0; a < 3; a++) {
                t[a] = lut ? build_tab_lut(rgl.n[a], gg.n[a]) : build_tab_generic(rgl.n[a], gvs[a]);
                dt[a] = h2d(t[a].data(), t[a].size()); wa.tab[a] = dt[a];
            }
            printf(" level %d: ref %dx%dx%d grid %dx%dx%d gvs=%g,%g,%g lut=%d\n", l, rgl.n[0], rgl.n[1], rgl.n[2], gg.n[0], gg.n[1], gg.n[2], gvs[0], gvs[1], gvs[2], (int)lut);
            float *dcpp = h2d(gp, grid->nvox);
            float *dwar = dalloc<float>(N), *dgr = dalloc<float>(3 * N), *ddef = dalloc<float>(3 * N);
            warp(wa, 1, dcpp, 0, lvf[l], 0, dwar, dgr, ddef, nullptr, st);
            CK(cudaStreamSynchronize(st));
            auto hd = d2h(ddef, 3 * N), hw = d2h(dwar, N), hgr = d2h(dgr, 3 * N);
            cmp("deformation field", hd.data(), (float *)def->data, 3 * N);
            cmp("warped image", hw.data(), (float *)war->data, N);
            cmp("warped gradient", hgr.data(), (float *)grad->data, 3 * N);
            // count of NaN
        }
    }
    return 0;
}
