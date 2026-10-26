// SPDX-License-Identifier: BSD-3-Clause
// NiftyReg reference run (reg_aladin_sym<float>, 2 levels to perform by default, rigid + affine, block step 1, align centre) that prints
// the forward matrix after every iteration ("it N type T: ...") and dumps the image pyramids (/tmp/dump/cpu_{tgt,tmpl}_L*.f32).
// usage: cpu_trace <target.nii[.gz]> <template.nii> [levels_to_perform=2]  -- takes ~30 s with 4 threads.  Compare with build/aladin_cli run with ALADIN_LP=2.
#include "_reg_aladin_sym.h"
#include "_reg_tools.h"
#include "nifti1_io.h"
#include <cstdio>
class Dbg : public reg_aladin_sym<float> {
public:
  int it=0;
  virtual void InitialiseRegistration() {
    reg_aladin_sym<float>::InitialiseRegistration();
    for(unsigned l=0;l<this->LevelsToPerform;l++){
      char fn[256];
      nifti_image* a=this->ReferencePyramid[l]; nifti_image* b=this->FloatingPyramid[l];
      sprintf(fn,"/tmp/dump/cpu_tgt_L%u.f32",l); FILE*f=fopen(fn,"wb"); fwrite(a->data,4,a->nx*a->ny*a->nz,f); fclose(f);
      sprintf(fn,"/tmp/dump/cpu_tmpl_L%u.f32",l); f=fopen(fn,"wb"); fwrite(b->data,4,b->nx*b->ny*b->nz,f); fclose(f);
      printf("PYR %u tgt %d %d %d  tmpl %d %d %d\n", l, a->nx,a->ny,a->nz,b->nx,b->ny,b->nz);
      printf("sto tgt:"); for(int i=0;i<3;i++) for(int j=0;j<4;j++) printf(" %.9g", a->sto_xyz.m[i][j]); printf("\n");
    }
  }
  void UpdateTransformationMatrix(int type) {
    reg_aladin_sym<float>::UpdateTransformationMatrix(type);
    printf("it %d type %d T:", it++, type);
    for(int i=0;i<3;i++) for(int j=0;j<4;j++) printf(" %.5f", this->TransformationMatrix->m[i][j]);
    printf("\n");
  }
};
int main(int argc,char**argv){
  nifti_image *ref=nifti_image_read(argv[1],true), *flo=nifti_image_read(argv[2],true);
  nifti_image *mask=nifti_copy_nim_info(ref); mask->data=calloc(ref->nvox,sizeof(float)); float*mp=(float*)mask->data; for(size_t i=0;i<ref->nvox;i++) mp[i]=1.f; mask->datatype=NIFTI_TYPE_FLOAT32; mask->nbyper=4;
  Dbg *r=new Dbg;
  r->SetInputReference(ref); r->SetInputFloating(flo); r->SetInputMask(mask);
  r->SetLevelsToPerform(argc>3?atoi(argv[3]):2); r->SetPerformAffine(1); r->SetBlockStepSize(1); r->SetPerformRigid(1); r->SetAlignCentre(1); r->SetVerbose(false);
  r->Run();
  mat44*m=r->GetTransformationMatrix();
  for(int i=0;i<4;i++) printf("%.7f %.7f %.7f %.7f\n", m->m[i][0],m->m[i][1],m->m[i][2],m->m[i][3]);
}
