// SPDX-License-Identifier: BSD-3-Clause
// CPU reference with per-evaluation tracing (subclass of reg_f3d)
#include "_reg_f3d.h"
#include "test_config.h"
#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_tools.h"
#include <cstdio>
#include <cstdlib>
static nifti_image *load(const char *fn)
{
    nifti_image *im = nifti_image_read(fn, true);
    nifti_free_extensions(im);
    im->scl_inter = 0; im->scl_slope = 1;
    if (im->datatype != NIFTI_TYPE_FLOAT32) reg_tools_changeDatatype<float>(im);
    nifti_update_dims_from_array(im);
    return im;
}
class dbg_f3d : public reg_f3d<float>
{
public:
    int nev = 0;
    dbg_f3d(int a, int b) : reg_f3d<float>(a, b) {}
    void SetNG() {}
    double GetObjectiveFunctionValue() override
    {
        double v = reg_f3d<float>::GetObjectiveFunctionValue();
        printf("EVAL %d obj=%.17g sim=%.17g be=%.17g le=%.17g jac=%.17g\n", nev++, v, currentWMeasure, currentWBE, currentWLE, currentWJac);
        return v;
    }
    int ngrad = 0;
    void GetObjectiveFunctionGradient() override
    {
        reg_f3d<float>::GetObjectiveFunctionGradient();
        if (ngrad < 4) {
            char fn[128]; snprintf(fn, sizeof fn, "/tmp/cpu_grad_L%d_%d.bin", currentLevel, ngrad);
            FILE *f = fopen(fn, "wb"); fwrite(transformationGradient->data, 4, transformationGradient->nvox, f); fclose(f);
        }
        ngrad++;
    }
    float NormaliseGradient() override
    {
        float r = reg_f3d<float>::NormaliseGradient();
        printf("NORM maxlen=%.9g\n", r);
        return r;
    }
    void UpdateParameters(float s) override
    {
        reg_f3d<float>::UpdateParameters(s);
        printf("UPD scale=%.9g\n", s);
    }
};
int main(int argc, char **argv)
{
    nifti_image *ref = load(argv[1]);
    nifti_image *flo = load(argv[2]);
    mat44 aff; reg_tool_ReadAffineFile(&aff, argv[3]);
    nifti_image *mask = nifti_copy_nim_info(ref);
    mask->data = calloc(ref->nvox, sizeof(float));
    for (size_t i = 0; i < ref->nvox; i++) ((float *)mask->data)[i] = 1.f;
    dbg_f3d *F = new dbg_f3d(ref->nt, flo->nt);
    F->SetReferenceImage(ref); F->SetFloatingImage(flo);
    F->DoNotPrintOutInformation();
    F->SetReferenceMask(mask); F->SetAffineTransformation(&aff);
    F->SetJacobianLogWeight(testcfg::jlWeight_()); F->SetBendingEnergyWeight(testcfg::beWeight_());
    F->SetMaximalIterationNumber(testcfg::maxIterations); F->SetLevelNumber(testcfg::levelNumber); F->SetLevelToPerform(testcfg::levelToPerform);
    F->SetSpacing(0, testcfg::spacingVox);
    F->UseLNCC(0, testcfg::lnccSigmaMM);
    F->UseLinearInterpolation();
    F->Run();
    nifti_image *cpp = F->GetControlPointPositionImage();
    if (argc > 4) { nifti_set_filenames(cpp, argv[4], 0, 0); nifti_image_write(cpp); }
    return 0;
}
