// SPDX-License-Identifier: BSD-3-Clause
// Compares the intermediates of one GPU iteration (dumped with ALADIN_DUMP=<dir> ALADIN_DUMP_IT=<it>) with the NiftyReg CPU
// functions fed with the same inputs: warped image, block matching correspondences and the LTS optimisation.
// usage: stage_check <target.nii[.gz]> <template.nii> <dumpdir> <it> <dir 0|1> <level> [1 = rigid iteration]
// CPU-side stage check using NiftyReg library: compares GPU intermediates at one iteration.
#include "AladinContent.h"
#include "_reg_blockMatching.h"
#include "_reg_globalTrans.h"
#include "_reg_resampling.h"
#include "_reg_tools.h"
#include "nifti1_io.h"
#include <cstdio>
#include <vector>
#include <cmath>
static std::vector<float> rd(const char* fn, size_t n) { std::vector<float> v(n); FILE* f = fopen(fn, "rb"); if (!f || fread(v.data(), 4, n, f) != n) { printf("read fail %s\n", fn); exit(1);} fclose(f); return v; }
int main(int argc, char** argv) {
  // args: target.nii template.nii dumpdir it dir(0/1)
  const char* dd = argv[3]; int it = atoi(argv[4]); int dir = atoi(argv[5]);
  nifti_image* ref0 = nifti_image_read(argv[1], true);
  nifti_image* flo0 = nifti_image_read(argv[2], true);
  nifti_image* refP[2]; nifti_image* floP[2];
  reg_createImagePyramid<float>(ref0, refP, 3, 2);
  reg_createImagePyramid<float>(flo0, floP, 3, 2);
  int L = argc > 6 ? atoi(argv[6]) : 1;
  nifti_image* R = dir == 0 ? refP[L] : floP[L];
  nifti_image* F = dir == 0 ? floP[L] : refP[L];
  char fn[512];
  sprintf(fn, "%s/it%d_%s.bin", dd, it, dir == 0 ? "T" : "B");
  std::vector<float> Tm = rd(fn, 16);
  mat44 T; for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) T.m[i][j] = Tm[i * 4 + j];
  size_t rv = (size_t)R->nx * R->ny * R->nz;
  int* mask = (int*)calloc(rv, sizeof(int));
  AladinContent con(R, F, mask, &T, sizeof(float), 50, 50, 1);
  _reg_blockMatchingParam* P = con.getBlockMatchingParams();
  printf("CPU: blocks total %d active %d\n", P->totalBlockNumber, P->activeBlockNumber);
  nifti_image* def = con.getCurrentDeformationField();
  nifti_image* warped = con.getCurrentWarped();
  reg_affine_getDeformationField(&T, def, false, mask);
  reg_resampleImage(F, warped, def, mask, 1, std::numeric_limits<float>::quiet_NaN(), NULL, NULL);
  std::vector<float> gw = rd((std::string(dd) + "/it" + std::to_string(it) + (dir == 0 ? "_warp0.bin" : "_warp1.bin")).c_str(), rv);
  float* cw = (float*)warped->data;
  size_t nanDiff = 0, cnt = 0; double maxd = 0, sumd = 0; size_t big = 0;
  for (size_t i = 0; i < rv; i++) {
    bool a = std::isnan(cw[i]), b = std::isnan(gw[i]);
    if (a != b) nanDiff++;
    else if (!a) { double d = fabs(cw[i] - gw[i]); cnt++; sumd += d; if (d > maxd) maxd = d; if (d > 0.01) big++; }
  }
  printf("WARP: nan-mismatch %zu of %zu, valid %zu, max|d| %.4g mean|d| %.4g, n>0.01: %zu\n", nanDiff, rv, cnt, maxd, sumd / cnt, big);
  // block matching on the GPU warped image (same input)
  memcpy(warped->data, gw.data(), rv * 4);
  block_matching_method(R, warped, P, mask);
  int K = P->activeBlockNumber;
  std::vector<float> gr = rd((std::string(dd) + "/it" + std::to_string(it) + (dir == 0 ? "_bmref0.bin" : "_bmref1.bin")).c_str(), (size_t)K * 3);
  std::vector<float> gp = rd((std::string(dd) + "/it" + std::to_string(it) + (dir == 0 ? "_bmwarp0.bin" : "_bmwarp1.bin")).c_str(), (size_t)K * 3);
  int refMis = 0, wMis = 0, nanMis = 0, both = 0;
  double maxDisp = 0;
  for (int i = 0; i < K; i++) {
    bool cn = std::isnan(P->warpedPosition[3 * i]), gn = std::isnan(gp[3 * i]);
    if (cn != gn) { nanMis++; continue; }
    if (fabs(P->referencePosition[3 * i] - gr[3 * i]) > 1e-3 || fabs(P->referencePosition[3 * i + 1] - gr[3 * i + 1]) > 1e-3 || fabs(P->referencePosition[3 * i + 2] - gr[3 * i + 2]) > 1e-3) refMis++;
    if (cn) continue;
    both++;
    double d = 0; for (int c = 0; c < 3; c++) d = fmax(d, fabs(P->warpedPosition[3 * i + c] - gp[3 * i + c]));
    if (d > 1e-3) { wMis++; if (d > maxDisp) maxDisp = d; if (wMis <= 6) printf("  mismatch rank %d ref (%.1f %.1f %.1f) cpu warp-ref (%.1f %.1f %.1f) gpu (%.1f %.1f %.1f)\n", i, gr[3*i],gr[3*i+1],gr[3*i+2], P->warpedPosition[3*i]-P->referencePosition[3*i], P->warpedPosition[3*i+1]-P->referencePosition[3*i+1], P->warpedPosition[3*i+2]-P->referencePosition[3*i+2], gp[3*i]-gr[3*i], gp[3*i+1]-gr[3*i+1], gp[3*i+2]-gr[3*i+2]); }
  }
  printf("BM: K=%d defined(cpu) %d refpos-mismatch %d, warped-pos mismatch %d of %d (max %.3f mm), nan mismatch %d\n", K, P->definedActiveBlockNumber, refMis, wMis, both, maxDisp, nanMis);
  // optimisation on the CPU from the CPU's own BM output, compare with the GPU's LTS result
  {
    mat44 Tn = T;
    optimize(P, &Tn, !(argc > 7 && atoi(argv[7]) == 1));
    sprintf(fn, "%s/it%d_Mnew%d.bin", dd, it, dir);
    std::vector<float> gm = rd(fn, 16);
    double md = 0;
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) md = fmax(md, fabs(Tn.m[i][j] - gm[i * 4 + j]));
    printf("OPT: max |cpu-gpu| of optimised matrix = %.3g\n cpu:", md);
    for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) printf(" %.6f", Tn.m[i][j]);
    printf("\n gpu:");
    for (int i = 0; i < 12; i++) printf(" %.6f", gm[i]);
    printf("\n");
  }
  return 0;
}
