// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "f3d_internal.h"
#include "f3d.h"
#include "test_config.h"
#include "_reg_tools.h"
#include "_reg_localTrans.h"
#include "_reg_localTrans_jac.h"
#include "_reg_localTrans_regul.h"
#include "_reg_resampling.h"
#include "_reg_globalTrans.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_lncc.h"
#include "_reg_f3d.h"
#include <random>
#include <cstdio>
#include <limits>
using namespace f3d;

static F3DParams test_params()
{
    F3DParams P;
    P.beWeight = testcfg::beWeight_(); P.jlWeight = testcfg::jlWeight_();
    P.levelNumber = testcfg::levelNumber; P.levelToPerform = testcfg::levelToPerform; P.maxIterations = testcfg::maxIterations;
    P.lnccSigmaMM = testcfg::lnccSigmaMM; P.spacingVox = testcfg::spacingVox;
    return P;
}

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
// returns max abs diff
static double cmp(const char *name, const float *a, const float *b, size_t n, double *meanOut = nullptr)
{
    double mx = 0, sum = 0, amax = 0; size_t nbad = 0, nan = 0, cnt = 0;
    for (size_t i = 0; i < n; i++) {
        if (a[i] != a[i] || b[i] != b[i]) { if ((a[i] != a[i]) != (b[i] != b[i])) nbad++; nan += (a[i] != a[i]); continue; }
        double d = fabs((double)a[i] - b[i]); mx = d > mx ? d : mx; sum += d; cnt++;
        amax = fmax(amax, fabs((double)b[i]));
    }
    printf("  %-30s n=%zu maxabs=%.3g mean=%.3g (ref max |v|=%.3g) nanMismatch=%zu (nan=%zu)\n", name, n, mx, sum / (cnt ? cnt : 1), amax, nbad, nan);
    if (meanOut) *meanOut = sum / (cnt ? cnt : 1);
    return mx;
}
static nifti_image *make_def(nifti_image *ref)
{
    nifti_image *def = nifti_copy_nim_info(ref);
    def->dim[0] = def->ndim = 5; def->dim[4] = def->nt = 1; def->dim[5] = def->nu = 3;
    def->nvox = (size_t)ref->nx * ref->ny * ref->nz * 3;
    def->nbyper = 4; def->datatype = NIFTI_TYPE_FLOAT32; def->data = calloc(def->nvox, 4);
    return def;
}
static nifti_image *make_vol(nifti_image *ref)
{
    nifti_image *w = nifti_copy_nim_info(ref);
    w->datatype = NIFTI_TYPE_FLOAT32; w->nbyper = 4; w->scl_slope = 1; w->scl_inter = 0; w->data = calloc(w->nvox, 4);
    return w;
}
// grid spacing (mm) of the level `l` of 3 for NiftyReg's first level spacing -5 voxels with levelNumber 4
static void level_spacing(const ImgGeom &full, int l, float sp[3])
{
    // gridSpacing at level 0 = 5*d*2^3 ; each level halves
    for (int a = 0; a < 3; a++) {
        float s = 5.f * full.d[a];
        float g = s * powf(2.f, 3.f);
        for (int k = 0; k < l; k++) g = g / 2.0f;
        sp[a] = g;
    }
}
