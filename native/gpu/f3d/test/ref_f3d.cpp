// SPDX-License-Identifier: BSD-3-Clause
// CPU reference: runs NiftyReg reg_f3d<float> (LNCC, cubic B-spline, settings of test_config.h) on a target / template pair.
// usage: ref_f3d <target.nii[.gz]> <template.nii> <affine.txt> <out_cpp.nii> [verbose=1]
#include "_reg_f3d.h"
#include "test_config.h"
#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_tools.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
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
int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage\n"); return 1; }
    nifti_image *ref = load(argv[1]);
    nifti_image *flo = load(argv[2]);
    mat44 aff;
    reg_tool_ReadAffineFile(&aff, argv[3]);
    int verbose = argc > 5 ? atoi(argv[5]) : 1;
    nifti_image *mask = nifti_copy_nim_info(ref);
    mask->data = calloc(ref->nvox, sizeof(float));
    for (size_t i = 0; i < ref->nvox; i++) ((float *)mask->data)[i] = 1.f;
    reg_f3d<float> *F = new reg_f3d<float>(ref->nt, flo->nt);
    F->SetReferenceImage(ref);
    F->SetFloatingImage(flo);
    if (verbose) F->PrintOutInformation(); else F->DoNotPrintOutInformation();
    F->SetReferenceMask(mask);
    F->SetAffineTransformation(&aff);
    F->SetJacobianLogWeight(testcfg::jlWeight_());
    F->SetBendingEnergyWeight(testcfg::beWeight_());
    F->SetMaximalIterationNumber(testcfg::maxIterations);
    F->SetLevelNumber(testcfg::levelNumber);
    F->SetLevelToPerform(testcfg::levelToPerform);
    F->SetSpacing(0, testcfg::spacingVox);
    for (int tp = 0; tp < ref->nt; tp++) F->UseLNCC(tp, testcfg::lnccSigmaMM);
    F->UseLinearInterpolation();
    auto t0 = std::chrono::steady_clock::now();
    F->Run();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "F3D wall time %.2f s\n", s);
    nifti_image *cpp = F->GetControlPointPositionImage();
    nifti_set_filenames(cpp, argv[4], 0, 0);
    nifti_image_write(cpp);
    return 0;
}
