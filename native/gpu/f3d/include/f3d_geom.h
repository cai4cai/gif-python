// SPDX-License-Identifier: BSD-3-Clause
// Host-side geometry helpers that mirror NiftyReg v1.5.77 (float mat44 with double internals).
// Only the sform (sform_code>0) path is supported.
#pragma once
#include <cmath>
#include <cstring>
#include <cstdint>

namespace f3d {

struct M44 { float m[4][4]; };
struct M33 { float m[3][3]; };

inline M44 m44_eye() { M44 r; memset(&r, 0, sizeof r); for (int i = 0; i < 4; i++) r.m[i][i] = 1.f; return r; }

// reg_mat44_mul(A,B)
inline M44 m44_mul(const M44 &A, const M44 &B)
{
    M44 R;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            R.m[i][j] = (float)((double)A.m[i][0] * (double)B.m[0][j] + (double)A.m[i][1] * (double)B.m[1][j] +
                                (double)A.m[i][2] * (double)B.m[2][j] + (double)A.m[i][3] * (double)B.m[3][j]);
    return R;
}
// reg_mat44_mul(mat, float in[3], float out[3])
inline void m44_vec(const M44 &M, const float in[3], float out[3])
{
    for (int i = 0; i < 3; i++)
        out[i] = (float)((double)M.m[i][0] * (double)in[0] + (double)M.m[i][1] * (double)in[1] +
                         (double)M.m[i][2] * (double)in[2] + (double)M.m[i][3]);
}
// nifti_mat44_inverse
inline M44 m44_inv(const M44 &R)
{
    double r11 = R.m[0][0], r12 = R.m[0][1], r13 = R.m[0][2];
    double r21 = R.m[1][0], r22 = R.m[1][1], r23 = R.m[1][2];
    double r31 = R.m[2][0], r32 = R.m[2][1], r33 = R.m[2][2];
    double v1 = R.m[0][3], v2 = R.m[1][3], v3 = R.m[2][3];
    double deti = r11 * r22 * r33 - r11 * r32 * r23 - r21 * r12 * r33 + r21 * r32 * r13 + r31 * r12 * r23 - r31 * r22 * r13;
    if (deti != 0.0) deti = 1.0 / deti;
    M44 Q;
    Q.m[0][0] = deti * (r22 * r33 - r32 * r23);
    Q.m[0][1] = deti * (-r12 * r33 + r32 * r13);
    Q.m[0][2] = deti * (r12 * r23 - r22 * r13);
    Q.m[0][3] = deti * (-r12 * r23 * v3 + r12 * v2 * r33 + r22 * r13 * v3 - r22 * v1 * r33 - r32 * r13 * v2 + r32 * v1 * r23);
    Q.m[1][0] = deti * (-r21 * r33 + r31 * r23);
    Q.m[1][1] = deti * (r11 * r33 - r31 * r13);
    Q.m[1][2] = deti * (-r11 * r23 + r21 * r13);
    Q.m[1][3] = deti * (r11 * r23 * v3 - r11 * v2 * r33 - r21 * r13 * v3 + r21 * v1 * r33 + r31 * r13 * v2 - r31 * v1 * r23);
    Q.m[2][0] = deti * (r21 * r32 - r31 * r22);
    Q.m[2][1] = deti * (-r11 * r32 + r31 * r12);
    Q.m[2][2] = deti * (r11 * r22 - r21 * r12);
    Q.m[2][3] = deti * (-r11 * r22 * v3 + r11 * r32 * v2 + r21 * r12 * v3 - r21 * r32 * v1 - r31 * r12 * v2 + r31 * r22 * v1);
    Q.m[3][0] = Q.m[3][1] = Q.m[3][2] = 0.f;
    Q.m[3][3] = (deti == 0.0) ? 0.f : 1.f;
    return Q;
}

// Geometry of an image (voxel grid): dims, voxel size, sform and its inverse
struct ImgGeom {
    int n[3];
    float d[3];
    M44 xyz;  // sto_xyz
    M44 ijk;  // sto_ijk
    size_t nvox() const { return (size_t)n[0] * n[1] * n[2]; }
};

inline ImgGeom make_geom(int nx, int ny, int nz, float dx, float dy, float dz, const M44 &sto_xyz)
{
    ImgGeom g; g.n[0] = nx; g.n[1] = ny; g.n[2] = nz; g.d[0] = dx; g.d[1] = dy; g.d[2] = dz;
    g.xyz = sto_xyz; g.ijk = m44_inv(sto_xyz); return g;
}

// reg_downsampleImage header update. axes: which axes are downsampled
inline ImgGeom geom_downsample(const ImgGeom &in, const bool ax[3])
{
    ImgGeom o = in;
    for (int i = 0; i < 3; i++) {
        if (in.n[i] > 1 && ax[i]) o.n[i] = (int)std::ceil(in.n[i] / 2.0);
        if (in.d[i] > 0 && ax[i]) o.d[i] = in.d[i] * 2.0f;
        if (ax[i]) { o.xyz.m[0][i] *= 2.f; o.xyz.m[1][i] *= 2.f; o.xyz.m[2][i] *= 2.f; }
    }
    o.ijk = m44_inv(o.xyz);
    return o;
}
// which axes get downsampled for an image of this size (NiftyReg rule: skip if n/2 < 32)
inline void downsample_axes(const ImgGeom &g, bool ax[3])
{
    for (int i = 0; i < 3; i++) ax[i] = !((g.n[i] / 2) < 32);
}

// Control point grid geometry (nodes)
struct GridGeom {
    int n[3];
    float d[3];  // spacing in mm
    M44 xyz, ijk;
    size_t nnodes() const { return (size_t)n[0] * n[1] * n[2]; }
};

// reg_createControlPointGrid (sform branch); spacing in mm
inline GridGeom create_grid(const ImgGeom &ref, const float sp[3])
{
    GridGeom g;
    g.n[0] = (int)(std::ceil(ref.n[0] * ref.d[0] / sp[0]) + 3.f);
    g.n[1] = (int)(std::ceil(ref.n[1] * ref.d[1] / sp[1]) + 3.f);
    g.n[2] = (int)(std::ceil(ref.n[2] * ref.d[2] / sp[2]) + 3.f);
    g.d[0] = sp[0]; g.d[1] = sp[1]; g.d[2] = sp[2];
    float ratio[3] = {g.d[0] / ref.d[0], g.d[1] / ref.d[1], g.d[2] / ref.d[2]};
    M44 s; memset(&s, 0, sizeof s);
    for (int r = 0; r < 3; r++) {
        s.m[r][0] = ref.xyz.m[r][0] * ratio[0];
        s.m[r][1] = ref.xyz.m[r][1] * ratio[1];
        s.m[r][2] = ref.xyz.m[r][2] * ratio[2];
        s.m[r][3] = ref.xyz.m[r][3];
    }
    s.m[3][0] = ref.xyz.m[3][0]; s.m[3][1] = ref.xyz.m[3][1]; s.m[3][2] = ref.xyz.m[3][2]; s.m[3][3] = ref.xyz.m[3][3];
    float oi[3] = {-1.f, -1.f, -1.f}, orr[3];
    m44_vec(s, oi, orr);
    s.m[0][3] = orr[0]; s.m[1][3] = orr[1]; s.m[2][3] = orr[2];
    g.xyz = s; g.ijk = m44_inv(s);
    return g;
}

// reg_spline_refineControlPointGrid header update (referenceImage != NULL, sform branch)
inline GridGeom refine_grid_geom(const GridGeom &in, const ImgGeom &ref)
{
    GridGeom g = in;
    g.d[0] = in.d[0] / 2.0f; g.d[1] = in.d[1] / 2.0f; g.d[2] = in.d[2] / 2.0f;
    g.n[0] = (int)(std::ceil(ref.n[0] * ref.d[0] / g.d[0]) + 3.f);
    g.n[1] = (int)(std::ceil(ref.n[1] * ref.d[1] / g.d[1]) + 3.f);
    g.n[2] = (int)(std::ceil(ref.n[2] * ref.d[2] / g.d[2]) + 3.f);
    float ratio[3] = {g.d[0] / ref.d[0], g.d[1] / ref.d[1], g.d[2] / ref.d[2]};
    M44 s; memset(&s, 0, sizeof s);
    for (int r = 0; r < 3; r++) {
        s.m[r][0] = ref.xyz.m[r][0] * ratio[0];
        s.m[r][1] = ref.xyz.m[r][1] * ratio[1];
        s.m[r][2] = ref.xyz.m[r][2] * ratio[2];
        s.m[r][3] = ref.xyz.m[r][3];
    }
    s.m[3][0] = s.m[3][1] = s.m[3][2] = 0.f; s.m[3][3] = 1.f;
    float oi[3] = {-1.f, -1.f, -1.f}, orr[3];
    m44_vec(s, oi, orr);
    s.m[0][3] = orr[0]; s.m[1][3] = orr[1]; s.m[2][3] = orr[2];
    g.xyz = s; g.ijk = m44_inv(s);
    return g;
}

// 3x3 helpers
inline M33 m33_of(const M44 &A)
{
    M33 o; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) o.m[i][j] = A.m[i][j]; return o;
}

}  // namespace f3d
