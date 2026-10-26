// SPDX-License-Identifier: BSD-3-Clause
// reg_scores: registration quality scores (NCC, LNCC, NMI) of warped templates against a target, using NiftyReg's own
// measure classes (reg_lncc, reg_nmi) exactly as reg_measure / F3D call them.
//
//   reg_scores <target.nii[.gz]> <template_fmt> <mode> <transform> <label> [first last]
//     template_fmt : printf pattern with one %d (template index), e.g. /path/tpl_%d.nii
//     mode         : grid   -> transform = printf pattern with one %d of NiftyReg control point grid files
//                               (cubic B-spline grid that already contains the affine, as written by F3D / f3d_run)
//                    affine -> transform = text file with a 4x4 NiftyReg affine (reference -> floating, world mm)
//     label        : free text copied to the CSV rows
//   Writes CSV rows to stdout (header first):
//     label,target,template,nvalid,nvox,ncc,lncc,nmi,jac_min,jac_neg_frac
//
// The warped template is produced as F3D does (WarpFloatingImage): deformation field from the grid with
// reg_spline_getDeformationField(grid, def, mask, false, true), reg_resampleImage with linear interpolation (inter 1) and NaN
// padding. Measures are evaluated on target voxels where the warped template is defined (not NaN), full resolution.
//   NCC  : global Pearson correlation (same code as reg_measure -ncc)
//   LNCC : reg_lncc, Gaussian kernel std-dev 5 mm, mean of |local NCC| over the valid voxels
//   NMI  : reg_nmi (Parzen window, 64 bins + 4 padding bins, NiftyReg default), as reg_measure -nmi
// jac_min / jac_neg_frac: minimum Jacobian determinant of the B-spline grid over the target voxels and fraction <= 0
// (grid mode only; NaN in affine mode).
#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_resampling.h"
#include "_reg_tools.h"
#include "_reg_nmi.h"
#include "_reg_lncc.h"
#include "_reg_localTrans.h"
#include "_reg_localTrans_jac.h"
#include "_reg_globalTrans.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

static nifti_image *copy_image(nifti_image *src)
{
    nifti_image *c = nifti_copy_nim_info(src);
    c->data = malloc(src->nvox * src->nbyper);
    memcpy(c->data, src->data, src->nvox * src->nbyper);
    return c;
}

int main(int argc, char **argv)
{
    if (argc < 6) { fprintf(stderr, "usage: %s target template_fmt grid|affine transform label [first last]\n", argv[0]); return 1; }
    std::string tgtFn = argv[1], tplFmt = argv[2], mode = argv[3], trFmt = argv[4], label = argv[5];
    int first = argc > 7 ? atoi(argv[6]) : 1, last = argc > 7 ? atoi(argv[7]) : 20;
    bool gridMode = mode == "grid";
    if (!gridMode && mode != "affine") { fprintf(stderr, "mode must be grid or affine\n"); return 1; }

    nifti_image *ref = reg_io_ReadImageFile(tgtFn.c_str());
    if (!ref) { fprintf(stderr, "cannot read %s\n", tgtFn.c_str()); return 1; }
    reg_tools_changeDatatype<float>(ref);
    size_t nv = (size_t)ref->nx * ref->ny * ref->nz;
    std::string tname = tgtFn.substr(tgtFn.find_last_of('/') + 1);
    tname = tname.substr(0, tname.find(".nii"));
    std::vector<int> mask(nv);
    for (size_t i = 0; i < nv; i++) mask[i] = (int)i;

    mat44 aff;
    if (!gridMode) reg_tool_ReadAffineFile(&aff, (char *)trFmt.c_str());

    printf("label,target,template,nvalid,nvox,ncc,lncc,nmi,jac_min,jac_neg_frac\n");
    char buf[4096];
    for (int t = first; t <= last; t++) {
        snprintf(buf, sizeof buf, tplFmt.c_str(), t);
        nifti_image *flo = reg_io_ReadImageFile(buf);
        if (!flo) { fprintf(stderr, "cannot read %s\n", buf); return 1; }
        reg_tools_changeDatatype<float>(flo);

        nifti_image *def = nifti_copy_nim_info(ref);
        def->ndim = def->dim[0] = 5; def->nt = def->dim[4] = 1; def->nu = def->dim[5] = 3;
        def->nvox = nv * 3; def->datatype = NIFTI_TYPE_FLOAT32; def->nbyper = sizeof(float);
        def->data = calloc(def->nvox, sizeof(float));
        def->scl_slope = 1.f; def->scl_inter = 0.f;
        double jmin = std::numeric_limits<double>::quiet_NaN(), jneg = jmin;
        if (gridMode) {
            snprintf(buf, sizeof buf, trFmt.c_str(), t);
            nifti_image *cpp = reg_io_ReadImageFile(buf);
            if (!cpp) { fprintf(stderr, "cannot read grid %s\n", buf); return 1; }
            reg_spline_getDeformationField(cpp, def, mask.data(), false, true);
            nifti_image *jac = nifti_copy_nim_info(ref);
            jac->ndim = jac->dim[0] = 3; jac->nt = jac->dim[4] = 1; jac->nu = jac->dim[5] = 1;
            jac->nvox = nv; jac->datatype = NIFTI_TYPE_FLOAT32; jac->nbyper = sizeof(float);
            jac->data = calloc(nv, sizeof(float));
            reg_spline_GetJacobianMap(cpp, jac);
            float *jp = (float *)jac->data; size_t neg = 0; double mn = 1e30;
            for (size_t i = 0; i < nv; i++) { if (jp[i] <= 0) neg++; if (jp[i] < mn) mn = jp[i]; }
            jmin = mn; jneg = (double)neg / nv;
            nifti_image_free(jac); nifti_image_free(cpp);
        } else {
            reg_affine_getDeformationField(&aff, def, false, mask.data());
        }

        nifti_image *war = nifti_copy_nim_info(ref);
        war->ndim = war->dim[0] = flo->ndim; war->nt = war->dim[4] = flo->nt; war->nu = war->dim[5] = flo->nu;
        war->nvox = nv * flo->nt * flo->nu;
        war->cal_min = flo->cal_min; war->cal_max = flo->cal_max;
        war->scl_inter = flo->scl_inter; war->scl_slope = flo->scl_slope;
        war->datatype = flo->datatype; war->nbyper = flo->nbyper;
        war->data = malloc(war->nvox * war->nbyper);
        reg_resampleImage(flo, war, def, mask.data(), 1, std::numeric_limits<float>::quiet_NaN());
        nifti_image_free(def);

        // NCC (identical to reg_measure -ncc)
        double ncc;
        size_t nvalid = 0;
        {
            float *rp = (float *)ref->data, *wp = (float *)war->data;
            double mr = 0, mw = 0;
            for (size_t i = 0; i < nv; i++) if (rp[i] == rp[i] && wp[i] == wp[i]) { mr += rp[i]; mw += wp[i]; nvalid++; }
            mr /= nvalid; mw /= nvalid;
            double sr = 0, sw = 0, c = 0;
            for (size_t i = 0; i < nv; i++) if (rp[i] == rp[i] && wp[i] == wp[i]) {
                double a = rp[i] - mr, b = wp[i] - mw; sr += a * a; sw += b * b; c += a * b; }
            ncc = c / (sqrt(sr) * sqrt(sw));
        }
        // LNCC and NMI rescale their image arguments in place -> private copies
        double lncc, nmi;
        {
            nifti_image *r2 = copy_image(ref), *w2 = copy_image(war);
            reg_lncc lm;
            lm.SetTimepointWeight(0, 1.0);
            lm.InitialiseMeasure(r2, w2, mask.data(), w2, NULL, NULL);
            lncc = lm.GetSimilarityMeasureValue();
            nifti_image_free(r2); nifti_image_free(w2);
        }
        {
            nifti_image *r2 = copy_image(ref), *w2 = copy_image(war);
            reg_nmi nm;
            nm.SetTimepointWeight(0, 1.0);
            nm.InitialiseMeasure(r2, w2, mask.data(), w2, NULL, NULL);
            nmi = nm.GetSimilarityMeasureValue();
            nifti_image_free(r2); nifti_image_free(w2);
        }
        printf("%s,%s,%d,%zu,%zu,%.6f,%.6f,%.6f,%.6f,%.6g\n", label.c_str(), tname.c_str(), t, nvalid, nv, ncc, lncc, nmi, jmin, jneg);
        fflush(stdout);
        nifti_image_free(flo); nifti_image_free(war);
    }
    return 0;
}
