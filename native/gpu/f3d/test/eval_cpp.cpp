// SPDX-License-Identifier: BSD-3-Clause
// Evaluate the final-level (1/2 resolution) LNCC similarity of a control point grid with NiftyReg's own code.
// usage: eval_cpp target template cpp.nii [gridbin]   (gridbin: raw float array replacing the data of cpp.nii)
#include "test_common.h"
int main(int argc, char **argv)
{
    nifti_image *ref = load(argv[1]), *flo = load(argv[2]);
    nifti_image *grid = nifti_image_read(argv[3], true);
    if (argc > 4) {
        FILE *f = fopen(argv[4], "rb");
        size_t n = fread(grid->data, 4, grid->nvox, f);
        fclose(f);
        if (n != grid->nvox) { fprintf(stderr, "size mismatch\n"); return 1; }
    }
    nifti_image *rp[3], *fp[3];
    reg_createImagePyramid<float>(ref, rp, 4, 3);
    reg_createImagePyramid<float>(flo, fp, 4, 3);
    int l = 2;
    size_t N = (size_t)rp[l]->nx * rp[l]->ny * rp[l]->nz;
    nifti_image *def = make_def(rp[l]);
    int *mask = (int *)calloc(N, sizeof(int));
    nifti_image *war = make_vol(rp[l]);
    nifti_image *wgr = nifti_copy_nim_info(def); wgr->data = calloc(def->nvox, 4);
    nifti_image *vgr = nifti_copy_nim_info(def); vgr->data = calloc(def->nvox, 4);
    reg_lncc lncc;
    lncc.SetKernelStandardDeviation(0, 5.f);
    lncc.SetTimepointWeight(0, 1.0);
    lncc.InitialiseMeasure(rp[l], fp[l], mask, war, wgr, vgr, NULL);
    reg_spline_getDeformationField(grid, def, mask, false, true);
    reg_resampleImage(fp[l], war, def, mask, 1, std::numeric_limits<float>::quiet_NaN());
    double v = lncc.GetSimilarityMeasureValue();
    printf("%.9f\n", v);
    return 0;
}
