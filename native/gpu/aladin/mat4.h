// SPDX-License-Identifier: BSD-3-Clause
// Small double-precision matrix helpers usable on host and device.
// Used for: nifti_mat44_inverse semantics, matrix log/exp (replaces Eigen's MatrixFunctions
// in NiftyReg's reg_mat44_logm/expm) and reg_mat44_avg2.
//
// Two code paths: a generic 4x4 path (inverse scaling-and-squaring log, scaling-and-squaring
// exp) and an affine-specialised path (3x3 log/exp + translation series) which is ~10x cheaper
// and is what the device code uses.  Both are tested against Eigen (test/maths_check.cpp).
#pragma once
#include <cmath>
#include <cstring>

#ifdef __CUDACC__
#define ALADIN_HD __host__ __device__
#else
#define ALADIN_HD
#endif

namespace aladin {

// Row-major float 4x4 (same memory layout as NiftyReg / nifti mat44).
struct Mat44 {
  float m[4][4];
};

// ---------------------------------------------------------------- generic 4x4 (double)
struct D4 {
  double a[16];
};

ALADIN_HD inline D4 d4_eye() {
  D4 r;
  for (int i = 0; i < 16; ++i) r.a[i] = (i % 5 == 0) ? 1.0 : 0.0;
  return r;
}
ALADIN_HD inline D4 d4_from(const Mat44& f) {
  D4 r;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) r.a[i * 4 + j] = (double)f.m[i][j];
  return r;
}
ALADIN_HD inline Mat44 d4_to_float(const D4& d) {
  Mat44 r;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) r.m[i][j] = (float)d.a[i * 4 + j];
  return r;
}
ALADIN_HD inline D4 d4_mul(const D4& A, const D4& B) {
  D4 R;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      double s = 0;
      for (int k = 0; k < 4; ++k) s += A.a[i * 4 + k] * B.a[k * 4 + j];
      R.a[i * 4 + j] = s;
    }
  return R;
}
ALADIN_HD inline D4 d4_add(const D4& A, const D4& B, double sb = 1.0) {
  D4 R;
  for (int i = 0; i < 16; ++i) R.a[i] = A.a[i] + sb * B.a[i];
  return R;
}
ALADIN_HD inline D4 d4_scale(const D4& A, double s) {
  D4 R;
  for (int i = 0; i < 16; ++i) R.a[i] = A.a[i] * s;
  return R;
}
ALADIN_HD inline double d4_norm1(const D4& A) {
  double best = 0;
  for (int j = 0; j < 4; ++j) {
    double s = 0;
    for (int i = 0; i < 4; ++i) s += fabs(A.a[i * 4 + j]);
    if (s > best) best = s;
  }
  return best;
}
// General 4x4 inverse (Gauss-Jordan, partial pivoting).
ALADIN_HD inline D4 d4_inv(const D4& A) {
  double M[4][8];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      M[i][j] = A.a[i * 4 + j];
      M[i][4 + j] = (i == j) ? 1.0 : 0.0;
    }
  for (int c = 0; c < 4; ++c) {
    int p = c;
    double best = fabs(M[c][c]);
    for (int r = c + 1; r < 4; ++r)
      if (fabs(M[r][c]) > best) { best = fabs(M[r][c]); p = r; }
    if (p != c)
      for (int j = 0; j < 8; ++j) { double t = M[c][j]; M[c][j] = M[p][j]; M[p][j] = t; }
    double piv = M[c][c];
    if (piv == 0.0) piv = 1e-300;
    double ip = 1.0 / piv;
    for (int j = 0; j < 8; ++j) M[c][j] *= ip;
    for (int r = 0; r < 4; ++r) {
      if (r == c) continue;
      double f = M[r][c];
      if (f != 0.0)
        for (int j = 0; j < 8; ++j) M[r][j] -= f * M[c][j];
    }
  }
  D4 R;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) R.a[i * 4 + j] = M[i][4 + j];
  return R;
}
// Principal matrix square root by the Denman-Beavers iteration.
ALADIN_HD inline D4 d4_sqrtm(const D4& A) {
  D4 Y = A, Z = d4_eye();
  for (int it = 0; it < 60; ++it) {
    D4 Yi = d4_inv(Y), Zi = d4_inv(Z);
    D4 Yn = d4_scale(d4_add(Y, Zi), 0.5);
    D4 Zn = d4_scale(d4_add(Z, Yi), 0.5);
    double diff = d4_norm1(d4_add(Yn, Y, -1.0));
    Y = Yn;
    Z = Zn;
    if (diff <= 1e-15 * d4_norm1(Y)) break;
  }
  return Y;
}
// Principal matrix logarithm (inverse scaling and squaring + atanh series).
ALADIN_HD inline D4 d4_logm(const D4& A) {
  D4 X = A;
  int s = 0;
  const D4 I = d4_eye();
  while (d4_norm1(d4_add(X, I, -1.0)) > 0.1 && s < 60) {
    X = d4_sqrtm(X);
    ++s;
  }
  D4 S = d4_mul(d4_add(X, I, -1.0), d4_inv(d4_add(X, I, 1.0)));
  D4 S2 = d4_mul(S, S);
  D4 term = S, sum = S;
  for (int k = 3; k <= 41; k += 2) {
    term = d4_mul(term, S2);
    sum = d4_add(sum, d4_scale(term, 1.0 / k));
  }
  return d4_scale(sum, 2.0 * (double)((long long)1 << s));
}
// Matrix exponential (scaling and squaring + Taylor).
ALADIN_HD inline D4 d4_expm(const D4& A) {
  int s = 0;
  double nrm = d4_norm1(A);
  while (nrm > 0.25 && s < 60) { nrm *= 0.5; ++s; }
  D4 B = d4_scale(A, 1.0 / (double)((long long)1 << s));
  D4 term = d4_eye(), sum = d4_eye();
  for (int k = 1; k <= 20; ++k) {
    term = d4_scale(d4_mul(term, B), 1.0 / k);
    sum = d4_add(sum, term);
  }
  for (int i = 0; i < s; ++i) sum = d4_mul(sum, sum);
  return sum;
}

// ---------------------------------------------------------------- affine-specialised path
struct D3 {
  double a[9];
};
ALADIN_HD inline D3 d3_eye() {
  D3 r;
  for (int i = 0; i < 9; ++i) r.a[i] = (i % 4 == 0) ? 1.0 : 0.0;
  return r;
}
ALADIN_HD inline D3 d3_mul(const D3& A, const D3& B) {
  D3 R;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) R.a[i * 3 + j] = A.a[i * 3] * B.a[j] + A.a[i * 3 + 1] * B.a[3 + j] + A.a[i * 3 + 2] * B.a[6 + j];
  return R;
}
ALADIN_HD inline D3 d3_add(const D3& A, const D3& B, double sb = 1.0) {
  D3 R;
  for (int i = 0; i < 9; ++i) R.a[i] = A.a[i] + sb * B.a[i];
  return R;
}
ALADIN_HD inline D3 d3_scale(const D3& A, double s) {
  D3 R;
  for (int i = 0; i < 9; ++i) R.a[i] = A.a[i] * s;
  return R;
}
ALADIN_HD inline double d3_norm1(const D3& A) {
  double best = 0;
  for (int j = 0; j < 3; ++j) {
    double s = fabs(A.a[j]) + fabs(A.a[3 + j]) + fabs(A.a[6 + j]);
    if (s > best) best = s;
  }
  return best;
}
ALADIN_HD inline D3 d3_inv(const D3& A) {
  const double* a = A.a;
  double c00 = a[4] * a[8] - a[5] * a[7], c01 = a[5] * a[6] - a[3] * a[8], c02 = a[3] * a[7] - a[4] * a[6];
  double det = a[0] * c00 + a[1] * c01 + a[2] * c02;
  double id = 1.0 / det;
  D3 R;
  R.a[0] = c00 * id;
  R.a[1] = (a[2] * a[7] - a[1] * a[8]) * id;
  R.a[2] = (a[1] * a[5] - a[2] * a[4]) * id;
  R.a[3] = c01 * id;
  R.a[4] = (a[0] * a[8] - a[2] * a[6]) * id;
  R.a[5] = (a[2] * a[3] - a[0] * a[5]) * id;
  R.a[6] = c02 * id;
  R.a[7] = (a[1] * a[6] - a[0] * a[7]) * id;
  R.a[8] = (a[0] * a[4] - a[1] * a[3]) * id;
  return R;
}
ALADIN_HD inline D3 d3_sqrtm(const D3& A) {
  D3 Y = A, Z = d3_eye();
  for (int it = 0; it < 60; ++it) {
    D3 Yi = d3_inv(Y), Zi = d3_inv(Z);
    D3 Yn = d3_scale(d3_add(Y, Zi), 0.5);
    D3 Zn = d3_scale(d3_add(Z, Yi), 0.5);
    double diff = d3_norm1(d3_add(Yn, Y, -1.0));
    Y = Yn;
    Z = Zn;
    if (diff <= 1e-15 * d3_norm1(Y)) break;
  }
  return Y;
}
ALADIN_HD inline D3 d3_logm(const D3& A) {
  D3 X = A;
  int s = 0;
  const D3 I = d3_eye();
  while (d3_norm1(d3_add(X, I, -1.0)) > 0.1 && s < 60) {
    X = d3_sqrtm(X);
    ++s;
  }
  D3 S = d3_mul(d3_add(X, I, -1.0), d3_inv(d3_add(X, I, 1.0)));
  D3 S2 = d3_mul(S, S);
  D3 term = S, sum = S;
  for (int k = 3; k <= 41; k += 2) {
    term = d3_mul(term, S2);
    sum = d3_add(sum, d3_scale(term, 1.0 / k));
  }
  return d3_scale(sum, 2.0 * (double)((long long)1 << s));
}
ALADIN_HD inline D3 d3_expm(const D3& A) {
  int s = 0;
  double nrm = d3_norm1(A);
  while (nrm > 0.25 && s < 60) { nrm *= 0.5; ++s; }
  D3 B = d3_scale(A, 1.0 / (double)((long long)1 << s));
  D3 term = d3_eye(), sum = d3_eye();
  for (int k = 1; k <= 20; ++k) {
    term = d3_scale(d3_mul(term, B), 1.0 / k);
    sum = d3_add(sum, term);
  }
  for (int i = 0; i < s; ++i) sum = d3_mul(sum, sum);
  return sum;
}
ALADIN_HD inline void d3_mulv(const D3& A, const double* v, double* o) {
  for (int i = 0; i < 3; ++i) o[i] = A.a[i * 3] * v[0] + A.a[i * 3 + 1] * v[1] + A.a[i * 3 + 2] * v[2];
}

ALADIN_HD inline bool is_affine_form(const Mat44& m) {
  return m.m[3][0] == 0.f && m.m[3][1] == 0.f && m.m[3][2] == 0.f && m.m[3][3] == 1.f;
}

// log of [A t; 0 1] = [L w; 0 0],  L = log A,  w = V^-1 t,  V^-1 = sum_k B_k L^k / k!  (Bernoulli series)
// Returns false if the series is not safe (||L|| too large) -> caller falls back to the generic path.
ALADIN_HD inline bool aff_logm(const Mat44& m, D4& out) {
  D3 A;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) A.a[i * 3 + j] = (double)m.m[i][j];
  D3 L = d3_logm(A);
  if (!(d3_norm1(L) < 0.9)) return false;
  D3 L2 = d3_mul(L, L);
  D3 Vi = d3_eye();
  Vi = d3_add(Vi, L, -0.5);
  const double c[] = {1.0 / 12.0, -1.0 / 720.0, 1.0 / 30240.0, -1.0 / 1209600.0, 1.0 / 47900160.0,
                      -691.0 / 1307674368000.0, 1.0 / 74724249600.0, -3617.0 / 10670622842880000.0};
  D3 P = L2;
  for (int k = 0; k < 8; ++k) {
    Vi = d3_add(Vi, P, c[k]);
    P = d3_mul(P, L2);
  }
  double t[3] = {(double)m.m[0][3], (double)m.m[1][3], (double)m.m[2][3]}, w[3];
  d3_mulv(Vi, t, w);
  for (int i = 0; i < 16; ++i) out.a[i] = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) out.a[i * 4 + j] = L.a[i * 3 + j];
    out.a[i * 4 + 3] = w[i];
  }
  return true;
}
// exp of [L w; 0 0] = [e^L, V w; 0 1],  V = sum_k L^k/(k+1)!
ALADIN_HD inline bool aff_expm(const D4& m, D4& out) {
  if (m.a[12] != 0.0 || m.a[13] != 0.0 || m.a[14] != 0.0 || m.a[15] != 0.0) return false;
  D3 L;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) L.a[i * 3 + j] = m.a[i * 4 + j];
  double nl = d3_norm1(L);
  if (!(nl < 8.0)) return false;
  D3 E = d3_expm(L);
  double w[3] = {m.a[3], m.a[7], m.a[11]};
  D3 V = d3_eye(), P = d3_eye();
  double f = 1.0;
  for (int k = 1; k <= 22; ++k) {
    P = d3_mul(P, L);
    f *= (double)(k + 1);
    V = d3_add(V, P, 1.0 / f);
  }
  double vw[3];
  d3_mulv(V, w, vw);
  for (int i = 0; i < 16; ++i) out.a[i] = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) out.a[i * 4 + j] = E.a[i * 3 + j];
    out.a[i * 4 + 3] = vw[i];
  }
  out.a[15] = 1.0;
  return true;
}

// ---------------------------------------------------------------- NiftyReg-semantics wrappers
// nifti_mat44_inverse semantics (affine inverse computed in double, stored as float).
ALADIN_HD inline Mat44 nifti_inverse(const Mat44& R) {
  double r11 = R.m[0][0], r12 = R.m[0][1], r13 = R.m[0][2];
  double r21 = R.m[1][0], r22 = R.m[1][1], r23 = R.m[1][2];
  double r31 = R.m[2][0], r32 = R.m[2][1], r33 = R.m[2][2];
  double v1 = R.m[0][3], v2 = R.m[1][3], v3 = R.m[2][3];
  double deti = r11 * r22 * r33 - r11 * r32 * r23 - r21 * r12 * r33 + r21 * r32 * r13 + r31 * r12 * r23 - r31 * r22 * r13;
  if (deti != 0.0) deti = 1.0 / deti;
  Mat44 Q;
  Q.m[0][0] = (float)(deti * (r22 * r33 - r32 * r23));
  Q.m[0][1] = (float)(deti * (-r12 * r33 + r32 * r13));
  Q.m[0][2] = (float)(deti * (r12 * r23 - r22 * r13));
  Q.m[0][3] = (float)(deti * (-r12 * r23 * v3 + r12 * v2 * r33 + r22 * r13 * v3 - r22 * v1 * r33 - r32 * r13 * v2 + r32 * v1 * r23));
  Q.m[1][0] = (float)(deti * (-r21 * r33 + r31 * r23));
  Q.m[1][1] = (float)(deti * (r11 * r33 - r31 * r13));
  Q.m[1][2] = (float)(deti * (-r11 * r23 + r21 * r13));
  Q.m[1][3] = (float)(deti * (r11 * r23 * v3 - r11 * v2 * r33 - r21 * r13 * v3 + r21 * v1 * r33 + r31 * r13 * v2 - r31 * v1 * r23));
  Q.m[2][0] = (float)(deti * (r21 * r32 - r31 * r22));
  Q.m[2][1] = (float)(deti * (-r11 * r32 + r31 * r12));
  Q.m[2][2] = (float)(deti * (r11 * r22 - r21 * r12));
  Q.m[2][3] = (float)(deti * (-r11 * r22 * v3 + r11 * r32 * v2 + r21 * r12 * v3 - r21 * r32 * v1 - r31 * r12 * v2 + r31 * r22 * v1));
  Q.m[3][0] = Q.m[3][1] = Q.m[3][2] = 0.f;
  Q.m[3][3] = (deti == 0.0) ? 0.f : 1.f;
  return Q;
}

// reg_mat44_logm / reg_mat44_expm semantics: float in, double compute, float out.
ALADIN_HD inline Mat44 reg_logm(const Mat44& m, bool generic = false) {
  D4 o;
  if (!generic && is_affine_form(m) && aff_logm(m, o)) return d4_to_float(o);
  return d4_to_float(d4_logm(d4_from(m)));
}
ALADIN_HD inline Mat44 reg_expm(const Mat44& m, bool generic = false) {
  D4 o, in = d4_from(m);
  if (!generic && aff_expm(in, o)) return d4_to_float(o);
  return d4_to_float(d4_expm(in));
}

// reg_mat44_avg2: exp(0.5 * (log A + log B)), rows 3 of the logs zeroed, float rounding at each step.
ALADIN_HD inline Mat44 reg_avg2(const Mat44& A, const Mat44& B, bool generic = false) {
  Mat44 la = reg_logm(A, generic), lb = reg_logm(B, generic);
  for (int i = 0; i < 4; ++i) la.m[3][i] = lb.m[3][i] = 0.f;
  Mat44 s;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      float sum = (float)((double)la.m[i][j] + (double)lb.m[i][j]);
      s.m[i][j] = (float)((double)sum * 0.5);
    }
  return reg_expm(s, generic);
}

}  // namespace aladin
