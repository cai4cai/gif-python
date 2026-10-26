// SPDX-License-Identifier: BSD-3-Clause
// Compares mat4.h (reg_logm / reg_expm / reg_avg2 / nifti_inverse) with NiftyReg's Eigen-based implementations on random
// affine matrices of the kind produced by the registration.  Build (Makefile target maths_check) links the static
// NiftyReg libraries.
#include <cstdio>
#include <cstdlib>
#include <random>
#include "_reg_maths.h"
#include "_reg_maths_eigen.h"
#include "nifti1_io.h"
#include "../mat4.h"
static aladin::Mat44 rnd(std::mt19937& g, bool rigid) {
  std::uniform_real_distribution<float> u(-1, 1);
  float a = 0.35f * u(g), b = 0.35f * u(g), c = 0.35f * u(g);
  float ca = cosf(a), sa = sinf(a), cb = cosf(b), sb = sinf(b), cc = cosf(c), sc = sinf(c);
  float R[3][3] = {{cb * cc, -cb * sc, sb}, {sa * sb * cc + ca * sc, -sa * sb * sc + ca * cc, -sa * cb}, {-ca * sb * cc + sa * sc, ca * sb * sc + sa * cc, ca * cb}};
  aladin::Mat44 m;
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) m.m[i][j] = (i == j);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) m.m[i][j] = R[i][j] * (rigid ? 1.f : (1.f + 0.12f * u(g))) + (rigid ? 0.f : 0.06f * u(g) * (i != j));
    m.m[i][3] = 80.f * u(g);
  }
  return m;
}
static mat44 to_nr(const aladin::Mat44& a) { mat44 m; for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) m.m[i][j] = a.m[i][j]; return m; }
struct Stat { int n = 0, same = 0; double maxd = 0; void add(const aladin::Mat44& a, const mat44& b) { for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) { ++n; if (a.m[i][j] == b.m[i][j]) ++same; maxd = fmax(maxd, fabs(a.m[i][j] - b.m[i][j])); } } };
int main() {
  std::mt19937 g(7);
  Stat sl, sl_gen, se, sa, si;
  for (int k = 0; k < 2000; ++k) {
    aladin::Mat44 A = rnd(g, k % 2), B = rnd(g, k % 3 == 0);
    mat44 a = to_nr(A), b = to_nr(B);
    sl.add(aladin::reg_logm(A), reg_mat44_logm(&a));
    sl_gen.add(aladin::reg_logm(A, true), reg_mat44_logm(&a));
    aladin::Mat44 L = aladin::reg_logm(A);
    mat44 l = to_nr(L);
    se.add(aladin::reg_expm(L), reg_mat44_expm(&l));
    sa.add(aladin::reg_avg2(A, B), reg_mat44_avg2(&a, &b));
    si.add(aladin::nifti_inverse(A), nifti_mat44_inverse(a));
  }
  auto pr = [](const char* n, const Stat& s) { printf("%-28s identical floats %6.2f%%   max |diff| %.3g\n", n, 100.0 * s.same / s.n, s.maxd); };
  pr("logm (affine fast path)", sl); pr("logm (generic 4x4 path)", sl_gen); pr("expm", se); pr("avg2", sa); pr("nifti_inverse", si);
  return 0;
}
