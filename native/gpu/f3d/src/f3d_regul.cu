// SPDX-License-Identifier: BSD-3-Clause
// Regularisation terms on the control point grid: bending energy, linear elasticity, Jacobian log penalty (approximated at the
// control points) and the folding correction -- values and gradients, mirroring NiftyReg v1.5.77 (3D, float, sform).
#include "f3d_internal.h"
#include "f3d_basis.h"
#include <cmath>
#include <algorithm>

#define FCH 2048   // chunk of the folded voxel compaction
namespace f3d {

// ------------------------------------------------------------------ constants: Jacobian-at-nodes basis (approximation)
__constant__ float c_jbX[27], c_jbY[27], c_jbZ[27];     // reg_cubic_spline_jacobian3D (approximation)
__constant__ float c_jgX[27], c_jgY[27], c_jgZ[27];     // reg_spline_jacobianDetGradient3D ("inverted on purpose")
static bool g_const_init = false;
static void init_constants()
{
    if (g_const_init) return;
    g_const_init = true;
    float normal[3] = {1.f / 6.f, 2.f / 3.f, 1.f / 6.f};
    float first[3] = {-0.5f, 0.f, 0.5f};
    float tempX[9], tempY[9], tempZ[9], bX[27], bY[27], bZ[27];
    int coord = 0;
    for (int c = 0; c < 3; c++)
        for (int b = 0; b < 3; b++) {
            tempX[coord] = normal[c] * normal[b];
            tempY[coord] = normal[c] * first[b];
            tempZ[coord] = first[c] * normal[b];
            coord++;
        }
    coord = 0;
    for (int bc = 0; bc < 9; bc++)
        for (int a = 0; a < 3; a++) {
            bX[coord] = tempX[bc] * first[a];
            bY[coord] = tempY[bc] * normal[a];
            bZ[coord] = tempZ[bc] * normal[a];
            coord++;
        }
    CK(cudaMemcpyToSymbol(c_jbX, bX, sizeof bX));
    CK(cudaMemcpyToSymbol(c_jbY, bY, sizeof bY));
    CK(cudaMemcpyToSymbol(c_jbZ, bZ, sizeof bZ));
    coord = 0;
    float gX[27], gY[27], gZ[27];
    for (int c = 2; c > -1; --c)
        for (int b = 2; b > -1; --b)
            for (int a = 2; a > -1; --a) {
                gX[coord] = normal[c] * normal[b] * first[a];
                gY[coord] = normal[c] * first[b] * normal[a];
                gZ[coord] = first[c] * normal[b] * normal[a];
                coord++;
            }
    CK(cudaMemcpyToSymbol(c_jgX, gX, sizeof gX));
    CK(cudaMemcpyToSymbol(c_jgY, gY, sizeof gY));
    CK(cudaMemcpyToSymbol(c_jgZ, gZ, sizeof gZ));
}

GridArgs make_grid_args(const GridGeom &g, const ImgGeom &ref)
{
    init_constants();
    GridArgs a;
    for (int i = 0; i < 3; i++) { a.n[i] = g.n[i]; a.d[i] = g.d[i]; a.rd[i] = ref.d[i]; }
    a.nn = g.nnodes();
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) { a.xyz[r * 3 + c] = g.xyz.m[r][c]; a.ijk[r * 3 + c] = g.ijk.m[r][c]; }
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 4; c++) a.lat[r * 4 + c] = g.xyz.m[r][c];
    return a;
}

// ------------------------------------------------------------------ small 3x3 helpers (nifti1_io.c semantics)
// nifti_mat33_determ / the determinant inside nifti_mat33_inverse, with the FMA structure GCC generated (aarch64)
__device__ __forceinline__ double m3_det_d(const float *R)
{
    double R00 = R[0], R01 = R[1], R02 = R[2], R10 = R[3], R11 = R[4], R12 = R[5], R20 = R[6], R21 = R[7], R22 = R[8];
    double t0 = R00 * R21;   // rounded products (no contraction)
    t0 = __dmul_rn(R00, R21);
    double t1 = __dmul_rn(t0, R12);
    double p2 = __dmul_rn(R00, R11);
    double d0 = __fma_rn(p2, R22, -t1);
    double d18 = __dmul_rn(R01, R10);
    d0 = __fma_rn(-d18, R22, d0);
    double d3 = __dmul_rn(R10, R21);
    d0 = __fma_rn(d3, R02, d0);
    double d2 = __dmul_rn(R01, R20);
    d0 = __fma_rn(d2, R12, d0);
    double d1 = __dmul_rn(R11, R20);
    d0 = __fma_rn(-d1, R02, d0);
    return d0;
}
__device__ __forceinline__ float m3_det(const float *R) { return (float)m3_det_d(R); }
__device__ __forceinline__ void m3_inv(const float *R, float *Q)
{
    double r11 = R[0], r12 = R[1], r13 = R[2], r21 = R[3], r22 = R[4], r23 = R[5], r31 = R[6], r32 = R[7], r33 = R[8];
    double deti = m3_det_d(R);
    if (deti != 0.0) deti = 1.0 / deti;
    Q[0] = (float)(deti * (r22 * r33 - r32 * r23));
    Q[1] = (float)(deti * (-r12 * r33 + r32 * r13));
    Q[2] = (float)(deti * (r12 * r23 - r22 * r13));
    Q[3] = (float)(deti * (-r21 * r33 + r31 * r23));
    Q[4] = (float)(deti * (r11 * r33 - r31 * r13));
    Q[5] = (float)(deti * (-r11 * r23 + r21 * r13));
    Q[6] = (float)(deti * (r21 * r32 - r31 * r22));
    Q[7] = (float)(deti * (-r11 * r32 + r31 * r12));
    Q[8] = (float)(deti * (r11 * r22 - r21 * r12));
}
__device__ __forceinline__ void m3_mul(const float *A, const float *B, float *C)
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) C[i * 3 + j] = A[i * 3 + 0] * B[0 * 3 + j] + A[i * 3 + 1] * B[1 * 3 + j] + A[i * 3 + 2] * B[2 * 3 + j];
}
__device__ __forceinline__ float m3_rownorm(const float *A)
{
    float r1 = (float)((double)fabsf(A[0]) + (double)fabsf(A[1]) + (double)fabsf(A[2]));
    float r2 = (float)((double)fabsf(A[3]) + (double)fabsf(A[4]) + (double)fabsf(A[5]));
    float r3 = (float)((double)fabsf(A[6]) + (double)fabsf(A[7]) + (double)fabsf(A[8]));
    if (r1 < r2) r1 = r2;
    if (r1 < r3) r1 = r3;
    return r1;
}
__device__ __forceinline__ float m3_colnorm(const float *A)
{
    float r1 = (float)((double)fabsf(A[0]) + (double)fabsf(A[3]) + (double)fabsf(A[6]));
    float r2 = (float)((double)fabsf(A[1]) + (double)fabsf(A[4]) + (double)fabsf(A[7]));
    float r3 = (float)((double)fabsf(A[2]) + (double)fabsf(A[5]) + (double)fabsf(A[8]));
    if (r1 < r2) r1 = r2;
    if (r1 < r3) r1 = r3;
    return r1;
}
// nifti_mat33_polar
__device__ void m3_polar(const float *A, float *Zout)
{
    float X[9], Y[9], Z[9];
    for (int i = 0; i < 9; i++) X[i] = A[i];
    float gam = m3_det(X), gmi, alp, bet, dif = 1.0f;
    int k = 0;
    while (gam == 0.0f) {
        gam = (float)(0.00001 * (0.001 + (double)m3_rownorm(X)));
        X[0] += gam; X[4] += gam; X[8] += gam;
        gam = m3_det(X);
    }
    while (true) {
        m3_inv(X, Y);
        if (dif > 0.3f) {
            alp = (float)sqrt((double)(m3_rownorm(X) * m3_colnorm(X)));
            bet = (float)sqrt((double)(m3_rownorm(Y) * m3_colnorm(Y)));
            gam = (float)sqrt((double)(bet / alp));
            gmi = (float)(1.0 / (double)gam);
        } else {
            gam = gmi = 1.0f;
        }
        Z[0] = 0.5f * __fmaf_rn(gmi, Y[0], __fmul_rn(gam, X[0]));
        Z[1] = 0.5f * __fmaf_rn(gmi, Y[3], __fmul_rn(gam, X[1]));
        Z[2] = 0.5f * __fmaf_rn(gmi, Y[6], __fmul_rn(gam, X[2]));
        Z[3] = 0.5f * __fmaf_rn(gmi, Y[1], __fmul_rn(gam, X[3]));
        Z[4] = 0.5f * __fmaf_rn(gmi, Y[4], __fmul_rn(gam, X[4]));
        Z[5] = 0.5f * __fmaf_rn(gmi, Y[7], __fmul_rn(gam, X[5]));
        Z[6] = 0.5f * __fmaf_rn(gmi, Y[2], __fmul_rn(gam, X[6]));
        Z[7] = 0.5f * __fmaf_rn(gmi, Y[5], __fmul_rn(gam, X[7]));
        Z[8] = 0.5f * __fmaf_rn(gmi, Y[8], __fmul_rn(gam, X[8]));
        double d = 0.0;
        for (int i = 0; i < 9; i++) d += fabs((double)(Z[i] - X[i]));
        dif = (float)d;
        k = k + 1;
        if (k > 100 || (double)dif < 3.e-6) break;
        for (int i = 0; i < 9; i++) X[i] = Z[i];
    }
    for (int i = 0; i < 9; i++) Zout[i] = Z[i];
}

// ---- fast numerics: the same 3x3 operations in plain float (no FP64: the FP64 pipe runs at 1/64 rate on this GPU)
__device__ __forceinline__ float m3f_det(const float *R)
{
    return __fmaf_rn(R[0], __fmaf_rn(R[4], R[8], -R[5] * R[7]), __fmaf_rn(-R[1], __fmaf_rn(R[3], R[8], -R[5] * R[6]), R[2] * __fmaf_rn(R[3], R[7], -R[4] * R[6])));
}
__device__ __forceinline__ void m3f_inv(const float *R, float *Q)
{
    float det = m3f_det(R);
    float di = det != 0.f ? __frcp_rn(det) : 0.f;
    Q[0] = di * __fmaf_rn(R[4], R[8], -R[7] * R[5]);
    Q[1] = di * __fmaf_rn(-R[1], R[8], R[7] * R[2]);
    Q[2] = di * __fmaf_rn(R[1], R[5], -R[4] * R[2]);
    Q[3] = di * __fmaf_rn(-R[3], R[8], R[6] * R[5]);
    Q[4] = di * __fmaf_rn(R[0], R[8], -R[6] * R[2]);
    Q[5] = di * __fmaf_rn(-R[0], R[5], R[3] * R[2]);
    Q[6] = di * __fmaf_rn(R[3], R[7], -R[6] * R[4]);
    Q[7] = di * __fmaf_rn(-R[0], R[7], R[6] * R[1]);
    Q[8] = di * __fmaf_rn(R[0], R[4], -R[3] * R[1]);
}
__device__ __forceinline__ float m3f_rownorm(const float *A)
{
    float r1 = fabsf(A[0]) + fabsf(A[1]) + fabsf(A[2]), r2 = fabsf(A[3]) + fabsf(A[4]) + fabsf(A[5]), r3 = fabsf(A[6]) + fabsf(A[7]) + fabsf(A[8]);
    return fmaxf(r1, fmaxf(r2, r3));
}
__device__ __forceinline__ float m3f_colnorm(const float *A)
{
    float r1 = fabsf(A[0]) + fabsf(A[3]) + fabsf(A[6]), r2 = fabsf(A[1]) + fabsf(A[4]) + fabsf(A[7]), r3 = fabsf(A[2]) + fabsf(A[5]) + fabsf(A[8]);
    return fmaxf(r1, fmaxf(r2, r3));
}
// Higham iteration of nifti_mat33_polar, float
__device__ void m3f_polar(const float *A, float *Zout)
{
    float X[9], Y[9], Z[9];
    for (int i = 0; i < 9; i++) X[i] = A[i];
    float gam = m3f_det(X), gmi, alp, bet, dif = 1.0f;
    int k = 0;
    while (gam == 0.0f) {
        gam = 0.00001f * (0.001f + m3f_rownorm(X));
        X[0] += gam; X[4] += gam; X[8] += gam;
        gam = m3f_det(X);
    }
    while (true) {
        m3f_inv(X, Y);
        if (dif > 0.3f) {
            alp = sqrtf(m3f_rownorm(X) * m3f_colnorm(X));
            bet = sqrtf(m3f_rownorm(Y) * m3f_colnorm(Y));
            gam = sqrtf(bet / alp);
            gmi = 1.f / gam;
        } else {
            gam = gmi = 1.0f;
        }
        Z[0] = 0.5f * __fmaf_rn(gmi, Y[0], gam * X[0]);
        Z[1] = 0.5f * __fmaf_rn(gmi, Y[3], gam * X[1]);
        Z[2] = 0.5f * __fmaf_rn(gmi, Y[6], gam * X[2]);
        Z[3] = 0.5f * __fmaf_rn(gmi, Y[1], gam * X[3]);
        Z[4] = 0.5f * __fmaf_rn(gmi, Y[4], gam * X[4]);
        Z[5] = 0.5f * __fmaf_rn(gmi, Y[7], gam * X[5]);
        Z[6] = 0.5f * __fmaf_rn(gmi, Y[2], gam * X[6]);
        Z[7] = 0.5f * __fmaf_rn(gmi, Y[5], gam * X[7]);
        Z[8] = 0.5f * __fmaf_rn(gmi, Y[8], gam * X[8]);
        float d = 0.f;
        for (int i = 0; i < 9; i++) d += fabsf(Z[i] - X[i]);
        dif = d;
        k = k + 1;
        if (k > 100 || dif < 3.e-6f) break;
        for (int i = 0; i < 9; i++) X[i] = Z[i];
    }
    for (int i = 0; i < 9; i++) Zout[i] = Z[i];
}

// test hooks: apply the 3x3 helpers to n matrices: out = {det, inv(9), polar(9), mul(A,A)(9), mul(inv(polar),A)}
__global__ void m3_test_k(const float *A, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float *a = A + 9 * i;
    float *o = out + 38 * i;
    o[0] = m3_det(a);
    m3_inv(a, o + 1);
    m3_polar(a, o + 10);
    m3_mul(a, a, o + 19);
    float R[9]; m3_inv(o + 10, R);
    m3_mul(R, a, o + 28);
    o[37] = 0.f;
}
void m3_test(const float *dA, float *dout, int n) { m3_test_k<<<cdiv(n, 128), 128>>>(dA, dout, n); }

// ------------------------------------------------------------------ block-wide double sum (all threads get the result)
__device__ double block_sum(double v, double *sh /* >= 32 */)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    double r = 0.0;
    if (threadIdx.x < 32) {
        r = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0;
        for (int o = 16; o > 0; o >>= 1) r += __shfl_down_sync(0xffffffffu, r, o);
        if (threadIdx.x == 0) sh[0] = r;
    }
    __syncthreads();
    r = sh[0];
    __syncthreads();
    return r;
}

// ------------------------------------------------------------------ Jacobian determinant at an interior node (approximation)
__device__ __forceinline__ void jac_node(const GridArgs &g, const float *cp, int x, int y, int z, float *J /*3x3 reoriented*/, float &det)
{
    const float *cx = cp, *cy = cp + g.nn, *cz = cp + 2 * g.nn;
    float m[9];
    for (int i = 0; i < 9; i++) m[i] = 0.f;
    int coord = 0;
    for (int c = 0; c < 3; c++)
        for (int b = 0; b < 3; b++)
            for (int a = 0; a < 3; a++) {
                size_t id = ((size_t)(z - 1 + c) * g.n[1] + (y - 1 + b)) * g.n[0] + (x - 1 + a);
                float vx = cx[id], vy = cy[id], vz = cz[id];
                // GCC (SLP vectorisation) left 4 of the 9 accumulations unfused (separate multiply and add): m00 m01 m02 m10
                m[0] = __fadd_rn(m[0], __fmul_rn(c_jbX[coord], vx));
                m[1] = __fadd_rn(m[1], __fmul_rn(c_jbY[coord], vx));
                m[2] = __fadd_rn(m[2], __fmul_rn(c_jbZ[coord], vx));
                m[3] = __fadd_rn(m[3], __fmul_rn(c_jbX[coord], vy));
                m[4] = __fmaf_rn(c_jbY[coord], vy, m[4]);
                m[5] = __fmaf_rn(c_jbZ[coord], vy, m[5]);
                m[6] = __fmaf_rn(c_jbX[coord], vz, m[6]);
                m[7] = __fmaf_rn(c_jbY[coord], vz, m[7]);
                m[8] = __fmaf_rn(c_jbZ[coord], vz, m[8]);
                coord++;
            }
    m3_mul(g.ijk, m, J);
    det = m3_det(J);
}

// compute all interior dets and matrices of every template. scratch layout per template: [det(nd)][J(9*nd)]
__global__ void jac_all_k(GridArgs g, const float *cp, size_t cpStride, float *scratch, size_t scrStride, const int *act)
{
    int b = blockIdx.y;
    if (act && !act[b]) return;
    int ndx = g.n[0] - 2, ndy = g.n[1] - 2, ndz = g.n[2] - 2;
    size_t nd = (size_t)ndx * ndy * ndz;
    float *det = scratch + (size_t)b * scrStride;
    float *Jm = det + nd;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nd; i += (size_t)gridDim.x * blockDim.x) {
        int x = i % ndx + 1, y = (i / ndx) % ndy + 1, z = i / ((size_t)ndx * ndy) + 1;
        float J[9], d;
        jac_node(g, cp + (size_t)b * cpStride, x, y, z, J, d);
        det[i] = d;
        for (int k = 0; k < 9; k++) Jm[i * 9 + k] = J[k];
    }
}

// ------------------------------------------------------------------ the evaluation kernel for one trial: parameter update,
// Jacobian penalty with folding correction, bending energy and linear energy values. One block per template.
__device__ double jac_penalty(const GridArgs &g, const float *cp, const float *det_unused, float *detS, float *Jm, bool needJ)
{
    // sums log(det)^2 over interior nodes, writes det/J arrays
    __shared__ double sh[32];
    int ndx = g.n[0] - 2, ndy = g.n[1] - 2, ndz = g.n[2] - 2;
    size_t nd = (size_t)ndx * ndy * ndz;
    double s = 0.0;
    for (size_t i = threadIdx.x; i < nd; i += blockDim.x) {
        int x = i % ndx + 1, y = (i / ndx) % ndy + 1, z = i / ((size_t)ndx * ndy) + 1;
        float J[9], d;
        jac_node(g, cp, x, y, z, J, d);
        detS[i] = d;
        if (needJ) for (int k = 0; k < 9; k++) Jm[i * 9 + k] = J[k];
        double ld = (double)logf(d);   // NiftyReg: log(float) -> logf
        s += ld * ld;
    }
    double tot = block_sum(s, sh);
    return tot / (double)nd;
}

// node (x,y,z) all nodes: folding correction using stored dets/matrices
__device__ bool fold_correct_node(const GridArgs &g, float *cp, const float *detS, const float *Jm, int x, int y, int z)
{
    bool changed = false;
    float fc[3] = {0.f, 0.f, 0.f};
    bool correct = false;
    int ndx = g.n[0] - 2, ndy = g.n[1] - 2;
    // basis at 0: values and first derivatives (get_BSplineBasisValue(0, idx, value, first))
    const double bv[3] = {1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0};
    const double bf[3] = {-0.5, 0.0, 0.5};
    for (int pz = z - 1; pz < z + 2; pz++) {
        if (pz > 0 && pz < g.n[2] - 1) {
            for (int py = y - 1; py < y + 2; py++) {
                if (py > 0 && py < g.n[1] - 1) {
                    for (int px = x - 1; px < x + 2; px++) {
                        if (px > 0 && px < g.n[0] - 1) {
                            size_t jid = ((size_t)(pz - 1) * ndy + (py - 1)) * ndx + (px - 1);
                            double detJac = detS[jid];
                            if (detJac <= 0.0) {
                                int iz = z - pz + 1, iy = y - py + 1, ix = x - px + 1;
                                float zB = (float)bv[iz], zF = (float)bf[iz], yB = (float)bv[iy], yF = (float)bf[iy],
                                      xB = (float)bv[ix], xF = (float)bf[ix];
                                float b0 = xF * yB * zB, b1 = xB * yF * zB, b2 = xB * yB * zF;
                                const float *J = Jm + jid * 9;
                                correct = true;
                                double dj = 1.0;
                                fc[0] = (float)((double)fc[0] + dj * (double)(b0 * (J[4] * J[8] - J[5] * J[7]) + b1 * (J[5] * J[6] - J[3] * J[8]) + b2 * (J[3] * J[7] - J[4] * J[6])));
                                fc[1] = (float)((double)fc[1] + dj * (double)(b0 * (J[2] * J[7] - J[1] * J[8]) + b1 * (J[0] * J[8] - J[2] * J[6]) + b2 * (J[1] * J[6] - J[0] * J[7])));
                                fc[2] = (float)((double)fc[2] + dj * (double)(b0 * (J[1] * J[5] - J[2] * J[4]) + b1 * (J[2] * J[3] - J[0] * J[5]) + b2 * (J[0] * J[4] - J[1] * J[3])));
                            }
                        }
                    }
                }
            }
        }
    }
    if (correct) {
        float gr[3];
        gr[0] = g.xyz[0] * fc[0] + g.xyz[1] * fc[1] + g.xyz[2] * fc[2];
        gr[1] = g.xyz[3] * fc[0] + g.xyz[4] * fc[1] + g.xyz[5] * fc[2];
        gr[2] = g.xyz[6] * fc[0] + g.xyz[7] * fc[1] + g.xyz[8] * fc[2];
        float norm = (float)(5.0 * (double)sqrtf(gr[0] * gr[0] + gr[1] * gr[1] + gr[2] * gr[2]));
        if (norm > 0.0f) {
            size_t id = ((size_t)z * g.n[1] + y) * g.n[0] + x;
            cp[id] += (float)(gr[0] / norm);
            cp[g.nn + id] += (float)(gr[1] / norm);
            cp[2 * g.nn + id] += (float)(gr[2] / norm);
            changed = true;
        }
    }
    return changed;
}

// bending energy contribution of one interior node (positions)
__device__ __forceinline__ double be_node_value(const GridArgs &g, const float *cp, int x, int y, int z)
{
    const float *sx = cp, *sy = cp + g.nn, *sz = cp + 2 * g.nn;
    float XX_x = 0, YY_x = 0, ZZ_x = 0, XY_x = 0, YZ_x = 0, XZ_x = 0;
    float XX_y = 0, YY_y = 0, ZZ_y = 0, XY_y = 0, YZ_y = 0, XZ_y = 0;
    float XX_z = 0, YY_z = 0, ZZ_z = 0, XY_z = 0, YZ_z = 0, XZ_z = 0;
    int k = 0;
    for (int c = -1; c < 2; c++)
        for (int b = -1; b < 2; b++)
            for (int a = -1; a < 2; a++) {
                size_t id = ((size_t)(z + c) * g.n[1] + (y + b)) * g.n[0] + (x + a);
                float cx = sx[id], cy = sy[id], cz = sz[id];
                XX_x += c_basis2XX[k] * cx; YY_x += c_basis2YY[k] * cx; ZZ_x += c_basis2ZZ[k] * cx;
                XY_x += c_basis2XY[k] * cx; YZ_x += c_basis2YZ[k] * cx; XZ_x += c_basis2XZ[k] * cx;
                XX_y += c_basis2XX[k] * cy; YY_y += c_basis2YY[k] * cy; ZZ_y += c_basis2ZZ[k] * cy;
                XY_y += c_basis2XY[k] * cy; YZ_y += c_basis2YZ[k] * cy; XZ_y += c_basis2XZ[k] * cy;
                XX_z += c_basis2XX[k] * cz; YY_z += c_basis2YY[k] * cz; ZZ_z += c_basis2ZZ[k] * cz;
                XY_z += c_basis2XY[k] * cz; YZ_z += c_basis2YZ[k] * cz; XZ_z += c_basis2XZ[k] * cz;
                ++k;
            }
    double v = (double)(XX_x * XX_x + YY_x * YY_x + ZZ_x * ZZ_x) + 2.0 * (double)(XY_x * XY_x + YZ_x * YZ_x + XZ_x * XZ_x);
    v += (double)(XX_y * XX_y); v += (double)(YY_y * YY_y); v += (double)(ZZ_y * ZZ_y);
    v += 2.0 * (double)(XY_y * XY_y + YZ_y * YZ_y + XZ_y * XZ_y);
    v += (double)(XX_z * XX_z); v += (double)(YY_z * YY_z); v += (double)(ZZ_z * ZZ_z);
    v += 2.0 * (double)(XY_z * XY_z + YZ_z * YZ_z + XZ_z * XZ_z);
    return v;
}
__device__ double be_value(const GridArgs &g, const float *cp)
{
    __shared__ double sh[32];
    int ndx = g.n[0] - 2, ndy = g.n[1] - 2, ndz = g.n[2] - 2;
    size_t nd = (size_t)ndx * ndy * ndz;
    double tot = 0.0;
    for (size_t i = threadIdx.x; i < nd; i += blockDim.x) {
        int x = i % ndx + 1, y = (i / ndx) % ndy + 1, z = i / ((size_t)ndx * ndy) + 1;
        tot += be_node_value(g, cp, x, y, z);
    }
    tot = block_sum(tot, sh);
    return tot / (double)(3 * g.nn);
}

// linear elasticity: per interior node matrix after polar decomposition, minus identity
__constant__ int c_le_mask = 0;
__constant__ int c_be_mask = 216096;     // neighbours a=0,+1: GCC's SLP loop version (only ZZ_z, XZ_*, YZ_y, YZ_z fused)
__constant__ int c_be_mask2 = 228839;    // neighbour a=-1: all fused except XY_x, XY_y, XY_z, YZ_x
void set_fuse_masks(int le, int be, int be2) { CK(cudaMemcpyToSymbol(c_le_mask, &le, sizeof le)); CK(cudaMemcpyToSymbol(c_be_mask, &be, sizeof be)); CK(cudaMemcpyToSymbol(c_be_mask2, &be2, sizeof be2)); }
__device__ __forceinline__ float acc_(float a, float b, float c, int fused) { return fused ? __fmaf_rn(a, b, c) : __fadd_rn(c, __fmul_rn(a, b)); }
template <bool GRAD>
__device__ void le_node(const GridArgs &g, const float *cp, int x, int y, int z, float *mat /*3x3 result*/)
{
    const float *sx = cp, *sy = cp + g.nn, *sz = cp + 2 * g.nn;
    float m[9];
    for (int i = 0; i < 9; i++) m[i] = 0.f;
    int i = 0;
    for (int c = -1; c < 2; c++)
        for (int b = -1; b < 2; b++)
            for (int a = -1; a < 2; a++) {
                size_t id = ((size_t)(z + c) * g.n[1] + (y + b)) * g.n[0] + (x + a);
                float cx = sx[id], cy = sy[id], cz = sz[id];
                const int fm = c_le_mask;
                m[0] = acc_(c_basis1X[i], cx, m[0], fm & 1); m[3] = acc_(c_basis1Y[i], cx, m[3], fm & 2); m[6] = acc_(c_basis1Z[i], cx, m[6], fm & 4);
                m[1] = acc_(c_basis1X[i], cy, m[1], fm & 8); m[4] = acc_(c_basis1Y[i], cy, m[4], fm & 16); m[7] = acc_(c_basis1Z[i], cy, m[7], fm & 32);
                m[2] = acc_(c_basis1X[i], cz, m[2], fm & 64); m[5] = acc_(c_basis1Y[i], cz, m[5], fm & 128);
                // gradient function (GCC): the last three products of every 9-block of m22 are fused
                m[8] = acc_(c_basis1Z[i], cz, m[8], (GRAD && (i % 9) >= 6) ? 1 : (fm & 256));
                ++i;
            }
    float t[9], P[9], R[9];
    m3_mul(g.ijk, m, t);
    if (g.fast) { m3f_polar(t, P); m3f_inv(P, R); }
    else { m3_polar(t, P); m3_inv(P, R); }
    m3_mul(R, t, mat);
    mat[0] -= 1.f; mat[4] -= 1.f; mat[8] -= 1.f;
}
__device__ __forceinline__ double le_node_value(const GridArgs &g, const float *cp, int x, int y, int z)
{
    float mat[9];
    le_node<false>(g, cp, x, y, z, mat);
    double cv = 0.;
    for (int b = 0; b < 3; b++)
        for (int a = 0; a < 3; a++) {
            double t = 0.5 * (double)(mat[a * 3 + b] + mat[b * 3 + a]);
            cv += t * t;
        }
    return cv;
}
__device__ double le_value(const GridArgs &g, const float *cp)
{
    __shared__ double sh[32];
    int ndx = g.n[0] - 2, ndy = g.n[1] - 2, ndz = g.n[2] - 2;
    size_t nd = (size_t)ndx * ndy * ndz;
    double tot = 0.0;
    for (size_t i = threadIdx.x; i < nd; i += blockDim.x) {
        int x = i % ndx + 1, y = (i / ndx) % ndy + 1, z = i / ((size_t)ndx * ndy) + 1;
        tot += le_node_value(g, cp, x, y, z);
    }
    tot = block_sum(tot, sh);
    return tot / (double)(3 * g.nn);
}

// One block per template: [param update] -> jacobian penalty (+correction loop) -> BE -> LE
// cur/best/grad: [B][3nn]; scratch: [B][scr] where scr >= nd*10 floats
__global__ void __launch_bounds__(1024) penalty_value_k(GridArgs g, float *cur, const float *best, size_t cpStride, float *scratch, size_t scrStride, float jlw,
                                PenaltyOut *out, const int *act, const int *fold, int *fix, int incr)
{
    // Folding: jacobian penalty with the correction loop of NiftyReg's ComputeJacobianBasedPenaltyTerm(1).
    // Only runs for templates whose first Jacobian penalty was NaN.
    int b = blockIdx.x;
    if (act && !act[b]) return;
    if (!fold[b]) return;
    float *cp = cur + (size_t)b * cpStride;
    const float *bp = best + (size_t)b * cpStride;
    size_t ndof = 3 * g.nn;
    size_t nd = (size_t)(g.n[0] - 2) * (g.n[1] - 2) * (g.n[2] - 2);
    float *detS = scratch + (size_t)b * scrStride;
    float *Jm = detS + nd;
    int restored = 0;
    double value;
    if (incr) {
        // Same result as the loop below, bit for bit, with far less work: (1) the dets / Jacobian matrices of the grid of the trial were
        // just written by pen_nodes_k (same grid, same arithmetic), so the first full evaluation is not repeated; (2) after a correction
        // pass only the dets that depend on a changed node are recomputed, only the nodes next to a folded det are visited (the other
        // nodes have no det <= 0 among their neighbours and are left unchanged by fold_correct_node anyway).
        extern __shared__ unsigned char sflag[];   // [nn] candidate nodes, [nd] dirty dets
        unsigned char *cand = sflag, *dirty = sflag + g.nn;
        __shared__ double sh[32];
        const int ndx = g.n[0] - 2, ndy = g.n[1] - 2;
        for (size_t i = threadIdx.x; i < g.nn + nd; i += blockDim.x) sflag[i] = 0;
        value = __longlong_as_double(0x7ff8000000000000ULL);   // NaN (fold[b] is only set when the penalty of this grid was NaN)
        int it = 0;
        const int maxit = 20;
        bool first = true;
        __syncthreads();
        while (value != value && it < maxit) {
            if (!first) {
                double s = 0.0;
                for (size_t i = threadIdx.x; i < nd; i += blockDim.x) {
                    float d;
                    if (dirty[i]) {
                        dirty[i] = 0;
                        int x = i % ndx + 1, y = (i / ndx) % ndy + 1, z = i / ((size_t)ndx * ndy) + 1;
                        float J[9];
                        jac_node(g, cp, x, y, z, J, d);
                        detS[i] = d;
                        for (int k = 0; k < 9; k++) Jm[i * 9 + k] = J[k];
                    } else d = detS[i];
                    double ld = (double)logf(d);
                    s += ld * ld;
                }
                value = block_sum(s, sh) / (double)nd;
            }
            first = false;
            if (value == value) { it++; break; }
            __syncthreads();
            // candidate nodes: all nodes that have a folded det (<= 0) among their neighbours
            for (size_t i = threadIdx.x; i < nd; i += blockDim.x) {
                if ((double)detS[i] <= 0.0) {
                    int x = i % ndx + 1, y = (i / ndx) % ndy + 1, z = i / ((size_t)ndx * ndy) + 1;
                    for (int c = -1; c < 2; c++)
                        for (int b2 = -1; b2 < 2; b2++)
                            for (int a = -1; a < 2; a++) cand[((size_t)(z + c) * g.n[1] + (y + b2)) * g.n[0] + (x + a)] = 1;
                }
            }
            __syncthreads();
            for (size_t i = threadIdx.x; i < g.nn; i += blockDim.x) {
                if (!cand[i]) continue;
                cand[i] = 0;
                int x = i % g.n[0], y = (i / g.n[0]) % g.n[1], z = i / ((size_t)g.n[0] * g.n[1]);
                if (fold_correct_node(g, cp, detS, Jm, x, y, z)) {
                    // the dets of the interior nodes next to this node depend on its position
                    for (int c = -1; c < 2; c++)
                        for (int b2 = -1; b2 < 2; b2++)
                            for (int a = -1; a < 2; a++) {
                                int px = x + a, py = y + b2, pz = z + c;
                                if (px > 0 && px < g.n[0] - 1 && py > 0 && py < g.n[1] - 1 && pz > 0 && pz < g.n[2] - 1)
                                    dirty[((size_t)(pz - 1) * ndy + (py - 1)) * ndx + (px - 1)] = 1;
                            }
                }
            }
            __syncthreads();
            value = __longlong_as_double(0x7ff8000000000000ULL);  // NaN: correctFolding returns NaN after correcting
            it++;
        }
    } else {
    value = jac_penalty(g, cp, nullptr, detS, Jm, true);
    int it = 0;
    const int maxit = 20;
    bool first = true;
    while (value != value && it < maxit) {
        if (!first) value = jac_penalty(g, cp, nullptr, detS, Jm, true);   // correctFolding recomputes the penalty first
        first = false;
        if (value == value) { it++; break; }
        __syncthreads();
        for (size_t i = threadIdx.x; i < g.nn; i += blockDim.x) {
            int x = i % g.n[0], y = (i / g.n[0]) % g.n[1], z = i / ((size_t)g.n[0] * g.n[1]);
            fold_correct_node(g, cp, detS, Jm, x, y, z);
        }
        __syncthreads();
        value = __longlong_as_double(0x7ff8000000000000ULL);  // NaN: correctFolding returns NaN after correcting
        it++;
    }
    }   // (old path)
    if (value != value) {
        for (size_t i = threadIdx.x; i < ndof; i += blockDim.x) cp[i] = bp[i];   // RestoreBestDOF
        __syncthreads();
        restored = 1;
    }
    if (threadIdx.x == 0) {
        out[b].jac = (double)jlw * value;
        out[b].restored = restored;
        fix[b] = restored ? 0 : 1;      // corrected successfully: bending / linear energy must be recomputed on the new grid
    }
}

__global__ void pen_update_k(size_t ndof, float *cur, const float *best, const float *grad, size_t cpStride, const float *scale, const int *act)
{
    int b = blockIdx.y;
    if (act && !act[b]) return;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ndof) return;
    cur[(size_t)b * cpStride + i] = best[(size_t)b * cpStride + i] + scale[b] * grad[(size_t)b * cpStride + i];
}
// node parallel evaluation of the three penalty sums (jacobian log^2, bending energy, linear energy)
__global__ void pen_nodes_k(GridArgs g, const float *cur, size_t cpStride, float *scratch, size_t scrStride, double *partial, int nblk,
                            float jlw, float bew, float lew, const int *act)
{
    int b = blockIdx.y;
    if (act && !act[b]) return;
    __shared__ double sh[32];
    int ndx = g.n[0] - 2, ndy = g.n[1] - 2, ndz = g.n[2] - 2;
    size_t nd = (size_t)ndx * ndy * ndz;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const float *cp = cur + (size_t)b * cpStride;
    double sj = 0.0, sb = 0.0, sl = 0.0;
    if (i < nd) {
        int x = i % ndx + 1, y = (i / ndx) % ndy + 1, z = i / ((size_t)ndx * ndy) + 1;
        if (jlw > 0.f) {
            float J[9], d;
            jac_node(g, cp, x, y, z, J, d);
            float *detS = scratch + (size_t)b * scrStride;
            detS[i] = d;
            for (int k = 0; k < 9; k++) detS[nd + i * 9 + k] = J[k];
            double ld = (double)logf(d);
            sj = ld * ld;
        }
        if (bew > 0.f) sb = be_node_value(g, cp, x, y, z);
        if (lew > 0.f) sl = le_node_value(g, cp, x, y, z);
    }
    sj = block_sum(sj, sh);
    sb = block_sum(sb, sh);
    sl = block_sum(sl, sh);
    if (threadIdx.x == 0) {
        double *p = partial + ((size_t)b * nblk + blockIdx.x) * 3;
        p[0] = sj; p[1] = sb; p[2] = sl;
    }
}
__global__ void pen_reduce_k(GridArgs g, const double *partial, int nblk, float jlw, float bew, float lew, PenaltyOut *out, int *fold, int *fix, const int *act)
{
    int b = blockIdx.x;
    if (act && !act[b]) return;
    __shared__ double sh[32];
    double sj = 0, sb = 0, sl = 0;
    for (int i = threadIdx.x; i < nblk; i += blockDim.x) {
        const double *p = partial + ((size_t)b * nblk + i) * 3;
        sj += p[0]; sb += p[1]; sl += p[2];
    }
    sj = block_sum(sj, sh); sb = block_sum(sb, sh); sl = block_sum(sl, sh);
    if (threadIdx.x == 0) {
        size_t nd = (size_t)(g.n[0] - 2) * (g.n[1] - 2) * (g.n[2] - 2);
        double jac = (jlw > 0.f) ? (double)jlw * (sj / (double)nd) : 0.0;
        out[b].jac = jac;
        out[b].be = (bew > 0.f) ? (double)bew * (sb / (double)(3 * g.nn)) : 0.0;
        out[b].le = (lew > 0.f) ? (double)lew * (sl / (double)(3 * g.nn)) : 0.0;
        out[b].restored = 0;
        fold[b] = (jac != jac) ? 1 : 0;
        fix[b] = 0;
    }
}

void penalty_bufs_init(PenaltyBufs &pb, const GridGeom &g, int B)
{
    size_t nd = (size_t)(g.n[0] - 2) * (g.n[1] - 2) * (g.n[2] - 2);
    pb.nblk = cdiv(nd, 128);
    pb.partial = dalloc<double>((size_t)B * pb.nblk * 3);
    pb.fold = dalloc_zero<int>(B);
    pb.fix = dalloc_zero<int>(B);
}
void penalty_bufs_free(PenaltyBufs &pb) { pool_free(pb.partial); pool_free(pb.fold); pool_free(pb.fix); }

__global__ void pen_reduce_bl_k(GridArgs g, const double *partial, int nblk, float bew, float lew, PenaltyOut *out, const int *act)
{
    int b = blockIdx.x;
    if (act && !act[b]) return;
    __shared__ double sh[32];
    double sb = 0, sl = 0, dummy = 0;
    for (int i = threadIdx.x; i < nblk; i += blockDim.x) {
        const double *p = partial + ((size_t)b * nblk + i) * 3;
        sb += p[1]; sl += p[2];
    }
    sb = block_sum(sb, sh); sl = block_sum(sl, sh);
    if (threadIdx.x == 0) {
        out[b].be = (bew > 0.f) ? (double)bew * (sb / (double)(3 * g.nn)) : 0.0;
        out[b].le = (lew > 0.f) ? (double)lew * (sl / (double)(3 * g.nn)) : 0.0;
    }
}

void penalty_value(const GridArgs &g, PenaltyBufs &pb, float *cur, const float *best, const float *grad, size_t cpStride,
                   float *scratch, size_t scrStride, const float *scale, float jlw, float bew, float lew, PenaltyOut *out, int B,
                   const int *act, cudaStream_t st)
{
    size_t ndof = 3 * g.nn;
    if (scale) pen_update_k<<<dim3(cdiv(ndof, 256), B), 256, 0, st>>>(ndof, cur, best, grad, cpStride, scale, act);
    pen_nodes_k<<<dim3(pb.nblk, B), 128, 0, st>>>(g, cur, cpStride, scratch, scrStride, pb.partial, pb.nblk, jlw, bew, lew, act);
    pen_reduce_k<<<B, 128, 0, st>>>(g, pb.partial, pb.nblk, jlw, bew, lew, out, pb.fold, pb.fix, act);
    if (jlw > 0.f) {
        // folding: single block correction loop for the templates whose Jacobian penalty is NaN
        static const bool noincr = getenv("F3D_NOFOLDINC") != nullptr;
        const size_t ndets = (size_t)(g.n[0] - 2) * (g.n[1] - 2) * (g.n[2] - 2);
        const size_t sm = g.nn + ndets;
        const int incr = (!noincr && sm <= 40000) ? 1 : 0;
        penalty_value_k<<<B, 1024, incr ? sm : 0, st>>>(g, cur, best, cpStride, scratch, scrStride, jlw, out, act, pb.fold, pb.fix, incr);
        // templates corrected successfully: recompute bending / linear energy on the corrected grid
        if (bew > 0.f || lew > 0.f) {
            pen_nodes_k<<<dim3(pb.nblk, B), 128, 0, st>>>(g, cur, cpStride, scratch, scrStride, pb.partial, pb.nblk, 0.f, bew, lew, pb.fix);
            pen_reduce_bl_k<<<B, 128, 0, st>>>(g, pb.partial, pb.nblk, bew, lew, out, pb.fix);
        }
    }
}


size_t penalty_scratch_floats(const GridGeom &g)
{
    size_t nd = (size_t)(g.n[0] - 2) * (g.n[1] - 2) * (g.n[2] - 2);
    size_t nn = g.nnodes();
    size_t a = nd * 10, b = nn * 21;
    return a > b ? a : b;
}

// ------------------------------------------------------------------ gradients
__device__ __forceinline__ void lat_init(const GridArgs &g, int x, int y, int z, float *init)
{
    // GCC (aarch64) hoists y*m1 out of the x loop: t = y*m1 (rounded); t = fma(x,m0,t); t = fma(m2,z,t); + m3
    for (int r = 0; r < 3; r++) {
        float t = __fmul_rn((float)y, g.lat[r * 4 + 1]);
        t = __fmaf_rn((float)x, g.lat[r * 4 + 0], t);
        t = __fmaf_rn(g.lat[r * 4 + 2], (float)z, t);
        init[r] = __fadd_rn(t, g.lat[r * 4 + 3]);
    }
}
__global__ void be_disp_k(GridArgs g, float *cp, float *dsp, size_t cpStride, size_t scrStride, const int *act)
{
    int b = blockIdx.y;
    if (act && !act[b]) return;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.nn) return;
    int x = i % g.n[0], y = (i / g.n[0]) % g.n[1], z = i / ((size_t)g.n[0] * g.n[1]);
    float init[3];
    lat_init(g, x, y, z, init);
    float *c = cp + (size_t)b * cpStride;
    float *d = dsp + (size_t)b * scrStride;
    for (int k = 0; k < 3; k++) d[k * g.nn + i] = c[k * g.nn + i] - init[k];
}
__global__ void be_deriv_k(GridArgs g, const float *dsp, float *D, size_t scrStride, const int *act)
{
    int b = blockIdx.y;
    if (act && !act[b]) return;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.nn) return;
    int x = i % g.n[0], y = (i / g.n[0]) % g.n[1], z = i / ((size_t)g.n[0] * g.n[1]);
    const float *d = dsp + (size_t)b * scrStride;
    float *o = D + (size_t)b * scrStride + 18 * i;
    float XX_x = 0, YY_x = 0, ZZ_x = 0, XY_x = 0, YZ_x = 0, XZ_x = 0;
    float XX_y = 0, YY_y = 0, ZZ_y = 0, XY_y = 0, YZ_y = 0, XZ_y = 0;
    float XX_z = 0, YY_z = 0, ZZ_z = 0, XY_z = 0, YZ_z = 0, XZ_z = 0;
    const bool interior = x > 0 && y > 0 && z > 0 && x < g.n[0] - 1 && y < g.n[1] - 1 && z < g.n[2] - 1;
    int k = 0;
    for (int c = -1; c < 2; c++)
        for (int b2 = -1; b2 < 2; b2++)
            for (int a = -1; a < 2; a++) {
                if (-1 < (x + a) && -1 < (y + b2) && -1 < (z + c) && (x + a) < g.n[0] && (y + b2) < g.n[1] && (z + c) < g.n[2]) {
                    size_t id = ((size_t)(z + c) * g.n[1] + (y + b2)) * g.n[0] + (x + a);
                    float cx = d[id], cy = d[g.nn + id], cz = d[2 * g.nn + id];
                    const int fm = (a == -1) ? c_be_mask2 : c_be_mask;
                    XX_x = acc_(c_basis2XX[k], cx, XX_x, fm & 1); YY_x = acc_(c_basis2YY[k], cx, YY_x, fm & 2); ZZ_x = acc_(c_basis2ZZ[k], cx, ZZ_x, fm & 4);
                    XY_x = acc_(c_basis2XY[k], cx, XY_x, fm & 8); YZ_x = acc_(c_basis2YZ[k], cx, YZ_x, fm & 16); XZ_x = acc_(c_basis2XZ[k], cx, XZ_x, fm & 32);
                    XX_y = acc_(c_basis2XX[k], cy, XX_y, fm & 64); YY_y = acc_(c_basis2YY[k], cy, YY_y, fm & 128); ZZ_y = acc_(c_basis2ZZ[k], cy, ZZ_y, fm & 256);
                    XY_y = acc_(c_basis2XY[k], cy, XY_y, fm & 512); YZ_y = acc_(c_basis2YZ[k], cy, YZ_y, fm & 1024); XZ_y = acc_(c_basis2XZ[k], cy, XZ_y, fm & 2048);
                    XX_z = acc_(c_basis2XX[k], cz, XX_z, fm & 4096); YY_z = acc_(c_basis2YY[k], cz, YY_z, fm & 8192); ZZ_z = acc_(c_basis2ZZ[k], cz, ZZ_z, fm & 16384);
                    XY_z = acc_(c_basis2XY[k], cz, XY_z, fm & 32768); YZ_z = acc_(c_basis2YZ[k], cz, YZ_z, fm & 65536); XZ_z = acc_(c_basis2XZ[k], cz, XZ_z, fm & 131072);
                }
                ++k;
            }
    o[0] = XX_x; o[1] = XX_y; o[2] = XX_z; o[3] = YY_x; o[4] = YY_y; o[5] = YY_z; o[6] = ZZ_x; o[7] = ZZ_y; o[8] = ZZ_z;
    o[9] = (float)(2.0 * XY_x); o[10] = (float)(2.0 * XY_y); o[11] = (float)(2.0 * XY_z);
    o[12] = (float)(2.0 * YZ_x); o[13] = (float)(2.0 * YZ_y); o[14] = (float)(2.0 * YZ_z);
    o[15] = (float)(2.0 * XZ_x); o[16] = (float)(2.0 * XZ_y); o[17] = (float)(2.0 * XZ_z);
}
__global__ void be_gather_k(GridArgs g, const float *D, float *ng, size_t cpStride, size_t scrStride, float approxRatio, const int *act)
{
    int b = blockIdx.y;
    if (act && !act[b]) return;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.nn) return;
    int x = i % g.n[0], y = (i / g.n[0]) % g.n[1], z = i / ((size_t)g.n[0] * g.n[1]);
    const float *Db = D + (size_t)b * scrStride;
    float gv[3] = {0.f, 0.f, 0.f};
    int a = 0;
    for (int Z = z - 1; Z < z + 2; Z++)
        for (int Y = y - 1; Y < y + 2; Y++)
            for (int X = x - 1; X < x + 2; X++) {
                if (-1 < X && -1 < Y && -1 < Z && X < g.n[0] && Y < g.n[1] && Z < g.n[2]) {
                    const float *p = Db + 18 * (((size_t)Z * g.n[1] + Y) * g.n[0] + X);
                    gv[0] += p[0] * c_basis2XX[a]; gv[1] += p[1] * c_basis2XX[a]; gv[2] += p[2] * c_basis2XX[a];
                    gv[0] += p[3] * c_basis2YY[a]; gv[1] += p[4] * c_basis2YY[a]; gv[2] += p[5] * c_basis2YY[a];
                    gv[0] += p[6] * c_basis2ZZ[a]; gv[1] += p[7] * c_basis2ZZ[a]; gv[2] += p[8] * c_basis2ZZ[a];
                    gv[0] += p[9] * c_basis2XY[a]; gv[1] += p[10] * c_basis2XY[a]; gv[2] += p[11] * c_basis2XY[a];
                    gv[0] += p[12] * c_basis2YZ[a]; gv[1] += p[13] * c_basis2YZ[a]; gv[2] += p[14] * c_basis2YZ[a];
                    gv[0] += p[15] * c_basis2XZ[a]; gv[1] += p[16] * c_basis2XZ[a]; gv[2] += p[17] * c_basis2XZ[a];
                }
                a++;
            }
    float *o = ng + (size_t)b * cpStride;
    o[i] += approxRatio * gv[0];
    o[g.nn + i] += approxRatio * gv[1];
    o[2 * g.nn + i] += approxRatio * gv[2];
}
__global__ void be_restore_k(GridArgs g, float *cp, const float *dsp, size_t cpStride, size_t scrStride, const int *act)
{
    int b = blockIdx.y;
    if (act && !act[b]) return;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.nn) return;
    int x = i % g.n[0], y = (i / g.n[0]) % g.n[1], z = i / ((size_t)g.n[0] * g.n[1]);
    float init[3];
    lat_init(g, x, y, z, init);
    float *c = cp + (size_t)b * cpStride;
    const float *d = dsp + (size_t)b * scrStride;
    for (int k = 0; k < 3; k++) c[k * g.nn + i] = d[k * g.nn + i] + init[k];
}
void be_gradient(const GridArgs &g, float *cp, float *nodeGrad, size_t cpStride, float *scratch, size_t scrStride, float weight,
                 int B, const int *act, cudaStream_t st)
{
    dim3 grid(cdiv(g.nn, 128), B);
    float *dsp = scratch, *D = scratch + 3 * g.nn;
    be_disp_k<<<grid, 128, 0, st>>>(g, cp, dsp, cpStride, scrStride, act);
    // derivative arrays need the displacement block and a separate region for D: dsp at scratch, D after it; both per-template strided
    be_deriv_k<<<grid, 128, 0, st>>>(g, dsp, D - 0, scrStride, act);
    float approxRatio = weight / (float)g.nn;
    be_gather_k<<<grid, 128, 0, st>>>(g, D, nodeGrad, cpStride, scrStride, approxRatio, act);
    be_restore_k<<<grid, 128, 0, st>>>(g, cp, dsp, cpStride, scrStride, act);
}

// linear elasticity
__global__ void le_diag_k(GridArgs g, const float *cp, float *diag, size_t cpStride, size_t scrStride, const int *act)
{
    int b = blockIdx.y;
    if (act && !act[b]) return;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.nn) return;
    int x = i % g.n[0], y = (i / g.n[0]) % g.n[1], z = i / ((size_t)g.n[0] * g.n[1]);
    float *o = diag + (size_t)b * scrStride + 3 * i;
    if (x < 1 || y < 1 || z < 1 || x > g.n[0] - 2 || y > g.n[1] - 2 || z > g.n[2] - 2) { o[0] = o[1] = o[2] = 0.f; return; }
    float mat[9];
    le_node<true>(g, cp + (size_t)b * cpStride, x, y, z, mat);
    o[0] = mat[0]; o[1] = mat[4]; o[2] = mat[8];
}
struct Mat9 { float v[9]; };
__global__ void le_gather_k(GridArgs g, const float *diag, float *ng, size_t cpStride, size_t scrStride, float approxRatio,
                            Mat9 invm, const int *act)
{
    int b = blockIdx.y;
    if (act && !act[b]) return;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.nn) return;
    int X = i % g.n[0], Y = (i / g.n[0]) % g.n[1], Z = i / ((size_t)g.n[0] * g.n[1]);
    const float *inv = invm.v;
    const float *dg = diag + (size_t)b * scrStride;
    float *o = ng + (size_t)b * cpStride;
    float ax = o[i], ay = o[g.nn + i], az = o[2 * g.nn + i];
    for (int c = 1; c >= -1; c--)
        for (int b2 = 1; b2 >= -1; b2--)
            for (int a = 1; a >= -1; a--) {
                int cx = X - a, cy = Y - b2, cz = Z - c;
                if (cx < 1 || cy < 1 || cz < 1 || cx > g.n[0] - 2 || cy > g.n[1] - 2 || cz > g.n[2] - 2) continue;
                int bi = 26 - ((c + 1) * 9 + (b2 + 1) * 3 + (a + 1));
                const float *p = dg + 3 * (((size_t)cz * g.n[1] + cy) * g.n[0] + cx);
                float gv0 = (float)(-2.0 * (double)p[0] * (double)c_basis1X[bi]);
                float gv1 = (float)(-2.0 * (double)p[1] * (double)c_basis1Y[bi]);
                float gv2 = (float)(-2.0 * (double)p[2] * (double)c_basis1Z[bi]);
                // GCC pattern (verified bit-exact against the library): t = g1*r1 (rounded); t = fma(g0,r0,t); t = fma(g2,r2,t);
                // out = fma(ratio, t, out)
                float t0 = __fmaf_rn(gv2, inv[2], __fmaf_rn(gv0, inv[0], __fmul_rn(gv1, inv[1])));
                float t1 = __fmaf_rn(gv2, inv[5], __fmaf_rn(gv0, inv[3], __fmul_rn(gv1, inv[4])));
                float t2 = __fmaf_rn(gv2, inv[8], __fmaf_rn(gv0, inv[6], __fmul_rn(gv1, inv[7])));
                ax = __fmaf_rn(approxRatio, t0, ax);
                ay = __fmaf_rn(approxRatio, t1, ay);
                az = __fmaf_rn(approxRatio, t2, az);
            }
    o[i] = ax; o[g.nn + i] = ay; o[2 * g.nn + i] = az;
}
void le_gradient(const GridArgs &g, const float *cp, float *nodeGrad, size_t cpStride, float *scratch, size_t scrStride, float weight,
                 int B, const int *act, cudaStream_t st)
{
    dim3 grid(cdiv(g.nn, 128), B);
    float *diag = scratch;
    le_diag_k<<<grid, 128, 0, st>>>(g, cp, diag, cpStride, scrStride, act);
    // inverse of reorientation (host, same arithmetic as nifti_mat33_inverse)
    double r11 = g.ijk[0], r12 = g.ijk[1], r13 = g.ijk[2], r21 = g.ijk[3], r22 = g.ijk[4], r23 = g.ijk[5], r31 = g.ijk[6], r32 = g.ijk[7], r33 = g.ijk[8];
    double deti = r11 * r22 * r33 - r11 * r32 * r23 - r21 * r12 * r33 + r21 * r32 * r13 + r31 * r12 * r23 - r31 * r22 * r13;
    if (deti != 0.0) deti = 1.0 / deti;
    float inv[9] = {(float)(deti * (r22 * r33 - r32 * r23)), (float)(deti * (-r12 * r33 + r32 * r13)), (float)(deti * (r12 * r23 - r22 * r13)),
                    (float)(deti * (-r21 * r33 + r31 * r23)), (float)(deti * (r11 * r33 - r31 * r13)), (float)(deti * (-r11 * r23 + r21 * r13)),
                    (float)(deti * (r21 * r32 - r31 * r22)), (float)(deti * (-r11 * r32 + r31 * r12)), (float)(deti * (r11 * r22 - r21 * r12))};
    Mat9 im; for (int k = 0; k < 9; k++) im.v[k] = inv[k];
    float approxRatio = weight / (float)g.nn;
    le_gather_k<<<grid, 128, 0, st>>>(g, diag, nodeGrad, cpStride, scrStride, approxRatio, im, act);
}

// Jacobian penalty gradient (approximation at the nodes)
__global__ void jac_grad_k(GridArgs g, const float *detS, const float *Jm_unused, float *ng, size_t cpStride, size_t scrStride,
                           float rx, float ry, float rz, const int *act)
{
    int b = blockIdx.y;
    if (act && !act[b]) return;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.nn) return;
    int x = i % g.n[0], y = (i / g.n[0]) % g.n[1], z = i / ((size_t)g.n[0] * g.n[1]);
    const float *det = detS + (size_t)b * scrStride;
    const float *Jmb = det + (size_t)(g.n[0] - 2) * (g.n[1] - 2) * (g.n[2] - 2);
    int ndx = g.n[0] - 2, ndy = g.n[1] - 2;
    float jc[3] = {0.f, 0.f, 0.f};
    int coord = 0;
    for (int pz = z - 1; pz < z + 2; ++pz) {
        if (pz > 0 && pz < g.n[2] - 1) {
            for (int py = y - 1; py < y + 2; ++py) {
                if (py > 0 && py < g.n[1] - 1) {
                    for (int px = x - 1; px < x + 2; ++px) {
                        if (px > 0 && px < g.n[0] - 1) {
                            size_t jid = ((size_t)(pz - 1) * ndy + (py - 1)) * ndx + (px - 1);
                            float detJac = det[jid];   // DTYPE detJac
                            if ((double)detJac > 0.0) {
                                const float *J = Jmb + jid * 9;
                                detJac = (float)(2.0 * (double)logf(detJac) / (double)detJac);
                                double dj = (double)detJac;
                                float b0 = c_jgX[coord], b1 = c_jgY[coord], b2 = c_jgZ[coord];
                                jc[0] = (float)((double)jc[0] + dj * (double)(b0 * (J[4] * J[8] - J[5] * J[7]) + b1 * (J[5] * J[6] - J[3] * J[8]) + b2 * (J[3] * J[7] - J[4] * J[6])));
                                jc[1] = (float)((double)jc[1] + dj * (double)(b0 * (J[2] * J[7] - J[1] * J[8]) + b1 * (J[0] * J[8] - J[2] * J[6]) + b2 * (J[1] * J[6] - J[0] * J[7])));
                                jc[2] = (float)((double)jc[2] + dj * (double)(b0 * (J[1] * J[5] - J[2] * J[4]) + b1 * (J[2] * J[3] - J[0] * J[5]) + b2 * (J[0] * J[4] - J[1] * J[3])));
                            }
                        }
                        coord++;
                    }
                } else coord += 3;
            }
        } else coord += 9;
    }
    float *o = ng + (size_t)b * cpStride;
    // GCC pattern: t = a0*b0 (rounded); t = fma(a1,b1,t); t = fma(a2,b2,t); out = fma(ratio,t,out)
    float t0 = __fmaf_rn(g.xyz[2], jc[2], __fmaf_rn(g.xyz[1], jc[1], __fmul_rn(g.xyz[0], jc[0])));
    float t1 = __fmaf_rn(g.xyz[5], jc[2], __fmaf_rn(g.xyz[4], jc[1], __fmul_rn(g.xyz[3], jc[0])));
    float t2 = __fmaf_rn(g.xyz[8], jc[2], __fmaf_rn(g.xyz[7], jc[1], __fmul_rn(g.xyz[6], jc[0])));
    o[i] = __fmaf_rn(rx, t0, o[i]);
    o[g.nn + i] = __fmaf_rn(ry, t1, o[g.nn + i]);
    o[2 * g.nn + i] = __fmaf_rn(rz, t2, o[2 * g.nn + i]);
}
void jac_gradient(const GridArgs &g, const float *cp, float *nodeGrad, size_t cpStride, float *scratch, size_t scrStride, float weight,
                  int B, const int *act, cudaStream_t st)
{
    size_t nd = (size_t)(g.n[0] - 2) * (g.n[1] - 2) * (g.n[2] - 2);
    dim3 gd(cdiv(nd, 128), B);
    jac_all_k<<<gd, 128, 0, st>>>(g, cp, cpStride, scratch, scrStride, act);
    float jn = (float)g.nn;
    float rx = g.rd[0] * weight / (jn * g.d[0]), ry = g.rd[1] * weight / (jn * g.d[1]), rz = g.rd[2] * weight / (jn * g.d[2]);
    dim3 grid(cdiv(g.nn, 128), B);
    jac_grad_k<<<grid, 128, 0, st>>>(g, scratch, nullptr, nodeGrad, cpStride, scrStride, rx, ry, rz, act);
}


// ------------------------------------------------------------------ voxel-level Jacobian
static void bspline_first(float basis, float v[4], float f[4])
{
    // get_BSplineBasisValues<float>(basis, values, first)
    float FF = basis * basis, FFF = FF * basis, MF = (float)(1.0 - basis);
    v[0] = (float)((double)(MF * MF * MF) / 6.0);
    v[1] = (float)((3.0 * FFF - 6.0 * FF + 4.0) / 6.0);
    v[2] = (float)((-3.0 * FFF + 3.0 * FF + 3.0 * basis + 1.0) / 6.0);
    v[3] = (float)(FFF / 6.0);
    f[3] = (float)(basis * basis / 2.0);
    f[0] = (float)(basis - 1.0 / 2.0 - f[3]);
    f[2] = (float)(1.0 + f[0] - 2.0 * f[3]);
    f[1] = -f[0] - f[2] - f[3];
}
std::vector<VoxTab> build_voxtab(int n, float gvs)
{
    std::vector<VoxTab> t(n);
    for (int i = 0; i < n; i++) {
        int pre = (int)((float)i / gvs);
        float basis = (float)i / gvs - (float)pre;
        if (basis < 0.f) basis = 0.f;
        t[i].pre = pre;
        bspline_first(basis, t[i].w, t[i].f);
    }
    return t;
}
void voxjac_init(VoxJacCtx &c, const GridArgs &ga, const GridGeom &g, const ImgGeom &ref, int Bc)
{
    c.Bc = Bc;
    c.ga = ga;
    for (int i = 0; i < 3; i++) { c.rn[i] = ref.n[i]; c.gvs[i] = g.d[i] / ref.d[i]; }
    c.N = ref.nvox();
    for (int a = 0; a < 3; a++) {
        auto t = build_voxtab(ref.n[a], c.gvs[a]);
        c.tab[a] = h2d(t.data(), t.size());
        std::vector<VoxTab2> t2(ref.n[a]);
        for (int p = 0; p < ref.n[a]; p++) {
            float basis = (float)p / c.gvs[a] - (float)t[p].pre;
            double bb = basis;
            t2[p].v[0] = (float)((1.0 - bb) * (1.0 - bb) * (1.0 - bb) / 6.0); t2[p].f[0] = (float)((2.0 * bb - bb * bb - 1.0) / 2.0);
            t2[p].v[1] = (float)((3.0 * bb * bb * bb - 6.0 * bb * bb + 4.0) / 6.0); t2[p].f[1] = (float)((3.0 * bb * bb - 4.0 * bb) / 2.0);
            t2[p].v[2] = (float)((3.0 * bb * bb - 3.0 * bb * bb * bb + 3.0 * bb + 1.0) / 6.0); t2[p].f[2] = (float)((2.0 * bb - 3.0 * bb * bb + 1.0) / 2.0);
            t2[p].v[3] = (float)(bb * bb * bb / 6.0); t2[p].f[3] = (float)(bb * bb / 2.0);
        }
        c.tab2[a] = h2d(t2.data(), t2.size());
    }
    c.det = dalloc<float>(c.N * Bc);
    c.J = dalloc<float>(9 * c.N * Bc);
    c.nblk = cdiv(c.N, 256);
    c.partial = dalloc<double>((size_t)c.nblk * Bc);
    c.rowFold = dalloc<unsigned char>((size_t)ref.n[1] * ref.n[2] * Bc);
    c.dact = dalloc<int>(Bc);
    c.diters = dalloc<int>(Bc);
    c.flist = dalloc<int>(c.N * Bc);
    c.fcnt = dalloc<int>((size_t)Bc * ((c.N + FCH - 1) / FCH));
    c.ftot = dalloc<int>(Bc);
    c.dirty = nullptr;
    if (!getenv("F3D_NODIRTY")) {
        for (int a = 0; a < 3; a++) c.cn[a] = (int)((float)(ref.n[a] - 1) / c.gvs[a]) + 1;
        c.ncell = (size_t)c.cn[0] * c.cn[1] * c.cn[2];
        c.dirty = dalloc<unsigned char>(c.ncell * Bc);
    }
}
void voxjac_free(VoxJacCtx &c)
{
    for (int a = 0; a < 3; a++) { pool_free(c.tab[a]); pool_free(c.tab2[a]); }
    pool_free(c.det); pool_free(c.J); pool_free(c.partial); pool_free(c.rowFold); pool_free(c.dact); pool_free(c.diters); pool_free(c.flist); pool_free(c.fcnt); pool_free(c.ftot); if (c.dirty) pool_free(c.dirty);
}

__global__ void voxjac_k(GridArgs g, int rx, int ry, int rz, const VoxTab *tx, const VoxTab *ty, const VoxTab *tz, const float *cp_,
                         float *det_, float *Jout_, double *partial_, size_t cpStride, const int *act, const unsigned char *dirty_, int cnx, int cny, size_t ncell)
{
    __shared__ double sh[32];
    const int bt = blockIdx.y;
    if (act && !act[bt]) return;
    size_t N = (size_t)rx * ry * rz;
    const float *cp = cp_ + (size_t)bt * cpStride;
    float *det = det_ + (size_t)bt * N;
    float *Jout = Jout_ ? Jout_ + (size_t)bt * 9 * N : nullptr;
    double *partial = partial_ + (size_t)bt * gridDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double s = 0.0;
    if (i < N) {
        int x = i % rx, y = (i / rx) % ry, z = i / ((size_t)rx * ry);
        VoxTab ax = tx[x], ay = ty[y], az = tz[z];
        if (dirty_ && !dirty_[(size_t)bt * ncell + ((size_t)az.pre * cny + ay.pre) * cnx + ax.pre]) {
            // none of the 64 control points of this voxel changed since the last evaluation: the determinant is still valid
            double ld = (double)logf(det[i]);
            s = ld * ld;
            goto done;
        }
        const float *cx = cp, *cy = cp + g.nn, *cz = cp + 2 * g.nn;
        float m[9];
        for (int k = 0; k < 9; k++) m[k] = 0.f;
        for (int c = 0; c < 4; c++) {
            int Z = min(max(az.pre + c, 0), g.n[2] - 1);
            for (int b = 0; b < 4; b++) {
                int Y = min(max(ay.pre + b, 0), g.n[1] - 1);
                float tX = az.w[c] * ay.w[b], tY = az.w[c] * ay.f[b], tZ = az.f[c] * ay.w[b];
                for (int a = 0; a < 4; a++) {
                    int X = min(max(ax.pre + a, 0), g.n[0] - 1);
                    size_t id = ((size_t)Z * g.n[1] + Y) * g.n[0] + X;
                    float bX = tX * ax.f[a], bY = tY * ax.w[a], bZ = tZ * ax.w[a];
                    float vx = cx[id], vy = cy[id], vz = cz[id];
                    m[0] += bX * vx; m[1] += bY * vx; m[2] += bZ * vx;
                    m[3] += bX * vy; m[4] += bY * vy; m[5] += bZ * vy;
                    m[6] += bX * vz; m[7] += bY * vz; m[8] += bZ * vz;
                }
            }
        }
        float J[9];
        m3_mul(g.ijk, m, J);
        float d = m3_det(J);
        det[i] = d;
        if (Jout && (double)d <= 0.0) for (int k = 0; k < 9; k++) Jout[i * 9 + k] = J[k];   // J is only read where det <= 0
        double ld = (double)logf(d);
        s = ld * ld;
    }
done:
    s = block_sum(s, sh);
    if (threadIdx.x == 0) partial[blockIdx.x] = s;
}
double voxjac_penalty(VoxJacCtx &c, const float *cp, cudaStream_t st)
{
    voxjac_k<<<c.nblk, 256, 0, st>>>(c.ga, c.rn[0], c.rn[1], c.rn[2], c.tab[0], c.tab[1], c.tab[2], cp, c.det, c.J, c.partial, 0, nullptr, nullptr, 0, 0, 0);
    auto h = d2h(c.partial, c.nblk);
    double s = 0.0;
    for (int i = 0; i < c.nblk; i++) s += h[i];
    return s / (double)c.N;
}

__global__ void foldrows_k(const float *det_, int rx, int ry, int rz, unsigned char *rowFold_, const int *act)
{
    const int bt = blockIdx.y;
    if (act && !act[bt]) return;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t N = (size_t)rx * ry * rz;
    if (i >= N) return;
    if ((double)det_[(size_t)bt * N + i] <= 0.0) rowFold_[(size_t)bt * ry * rz + i / rx] = 1;
}
__global__ void voxfold_k(GridArgs g, int rx, int ry, int rz, float g0, float g1, float g2, const VoxTab *tx, const VoxTab *ty,
                          const VoxTab *tz, const float *det_, const float *Jm_, float *cp_, const unsigned char *rowFold_, const VoxTab2 *t2x, const VoxTab2 *t2y, const VoxTab2 *t2z,
                          size_t cpStride, const int *act, unsigned char *dirty_, int cnx, int cny, int cnz, size_t ncell,
                          const int *flist_, const int *ftot)
{
    const int bt = blockIdx.y;
    if (act && !act[bt]) return;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.nn) return;
    const size_t Nv = (size_t)rx * ry * rz;
    const float *det = det_ + (size_t)bt * Nv;
    const float *Jm = Jm_ + (size_t)bt * 9 * Nv;
    float *cp = cp_ + (size_t)bt * cpStride;
    const unsigned char *rowFold = rowFold_ ? rowFold_ + (size_t)bt * ry * rz : nullptr;
    int x = i % g.n[0], y = (i / g.n[0]) % g.n[1], z = i / ((size_t)g.n[0] * g.n[1]);
    float fc[3] = {0.f, 0.f, 0.f};
    bool correct = false;
    int z0 = (int)ceilf((float)(z - 3) * g2), z1 = (int)floorf((float)(z + 1) * g2);
    int y0 = (int)ceilf((float)(y - 3) * g1), y1 = (int)floorf((float)(y + 1) * g1);
    int x0 = (int)ceilf((float)(x - 3) * g0), x1 = (int)floorf((float)(x + 1) * g0);
    // contribution of one folded voxel (px, py, pz) to the correction of this node
    auto body = [&](int px, int py, int pz, size_t jid) {
        const float *J = Jm + jid * 9;
        // basis values (value, first) of the node (x,y,z) at this voxel: index = node - pre
        int ix = x - tx[px].pre, iy = y - ty[py].pre, iz = z - tz[pz].pre;
        float xb = 0.f, xf = 0.f, yb = 0.f, yf = 0.f, zb = 0.f, zf = 0.f;
        if (ix >= 0 && ix < 4) { xb = t2x[px].v[ix]; xf = t2x[px].f[ix]; }
        if (iy >= 0 && iy < 4) { yb = t2y[py].v[iy]; yf = t2y[py].f[iy]; }
        if (iz >= 0 && iz < 4) { zb = t2z[pz].v[iz]; zf = t2z[pz].f[iz]; }
        float b0 = xf * yb * zb, b1 = xb * yf * zb, b2 = xb * yb * zf;
        correct = true;
        fc[0] = (float)((double)fc[0] + 1.0 * (double)(b0 * (J[4] * J[8] - J[5] * J[7]) + b1 * (J[5] * J[6] - J[3] * J[8]) + b2 * (J[3] * J[7] - J[4] * J[6])));
        fc[1] = (float)((double)fc[1] + 1.0 * (double)(b0 * (J[2] * J[7] - J[1] * J[8]) + b1 * (J[0] * J[8] - J[2] * J[6]) + b2 * (J[1] * J[6] - J[0] * J[7])));
        fc[2] = (float)((double)fc[2] + 1.0 * (double)(b0 * (J[1] * J[5] - J[2] * J[4]) + b1 * (J[2] * J[3] - J[0] * J[5]) + b2 * (J[0] * J[4] - J[1] * J[3])));
    };
    if (flist_) {
        // sorted list of the folded voxels (ascending linear index = the order of the pz, py, px loops below): only visit those of the box
        const int tot = ftot[bt];
        const int *fl = flist_ + (size_t)bt * Nv;
        const int zlo = max(z0, 0), zhi = min(z1, rz);
        if (zhi > zlo && tot > 0) {
            const int klo = zlo << 20, khi = zhi << 20;   // packed coordinates (pz << 20 | py << 10 | px): same order as the linear index
            int lo = 0, hi = tot;
            while (lo < hi) { int m = (lo + hi) >> 1; if (fl[m] < klo) lo = m + 1; else hi = m; }
            const int s0 = lo;
            hi = tot;
            while (lo < hi) { int m = (lo + hi) >> 1; if (fl[m] < khi) lo = m + 1; else hi = m; }
            for (int k = s0; k < lo; k++) {
                int v = fl[k];
                int px = v & 1023, py = (v >> 10) & 1023, pz = v >> 20;
                if (px >= x0 && px < x1 && py >= y0 && py < y1) body(px, py, pz, ((size_t)pz * ry + py) * rx + px);
            }
        }
    } else {
        for (int pz = z0; pz < z1; pz++) {
            if (pz > -1 && pz < rz) {
                for (int py = y0; py < y1; py++) {
                    if (py > -1 && py < ry && rowFold[(size_t)pz * ry + py]) {
                        for (int px = x0; px < x1; px++) {
                            if (px > -1 && px < rx) {
                                size_t jid = ((size_t)pz * ry + py) * rx + px;
                                double detJac = det[jid];
                                if (detJac <= 0.0) body(px, py, pz, jid);
                            }
                        }
                    }
                }
            }
        }
    }
    if (correct) {
        float gr[3];
        gr[0] = g.xyz[0] * fc[0] + g.xyz[1] * fc[1] + g.xyz[2] * fc[2];
        gr[1] = g.xyz[3] * fc[0] + g.xyz[4] * fc[1] + g.xyz[5] * fc[2];
        gr[2] = g.xyz[6] * fc[0] + g.xyz[7] * fc[1] + g.xyz[8] * fc[2];
        float norm = (float)(5.0 * (double)sqrtf(gr[0] * gr[0] + gr[1] * gr[1] + gr[2] * gr[2]));
        if (norm > 0.0f) {
            cp[i] += (float)(gr[0] / norm);
            cp[g.nn + i] += (float)(gr[1] / norm);
            cp[2 * g.nn + i] += (float)(gr[2] / norm);
            if (dirty_) {   // every cell whose (clamped) 4x4x4 control points contain this node needs a new determinant
                int xl = max(x - 3, 0), xh = (x == g.n[0] - 1) ? cnx - 1 : min(x, cnx - 1);
                int yl = max(y - 3, 0), yh = (y == g.n[1] - 1) ? cny - 1 : min(y, cny - 1);
                int zl = max(z - 3, 0), zh = (z == g.n[2] - 1) ? cnz - 1 : min(z, cnz - 1);
                for (int pz = zl; pz <= zh; pz++)
                    for (int py = yl; py <= yh; py++)
                        for (int px = xl; px <= xh; px++) dirty_[(size_t)bt * ncell + ((size_t)pz * cny + py) * cnx + px] = 1;
            }
        }
    }
}
// ---- order preserving compaction of the folded voxels (det <= 0) of every active template: count per chunk, scan, write
__global__ void fl_count_k(const float *det_, size_t N, int *cnt, int nch, const int *act)
{
    const int bt = blockIdx.y;
    if (act && !act[bt]) return;
    const float *det = det_ + (size_t)bt * N;
    size_t base = (size_t)blockIdx.x * FCH;
    int c = 0;
    for (int k = threadIdx.x; k < FCH; k += blockDim.x) { size_t i = base + k; if (i < N && (double)det[i] <= 0.0) c++; }
    __shared__ int sh[32];
    for (int o = 16; o > 0; o >>= 1) c += __shfl_down_sync(0xffffffffu, c, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = c;
    __syncthreads();
    if (threadIdx.x == 0) { int t = 0; for (int w = 0; w < (int)(blockDim.x >> 5); w++) t += sh[w]; cnt[(size_t)bt * nch + blockIdx.x] = t; }
}
__global__ void fl_scan_k(int *cnt, int nch, int *tot, const int *act)
{
    const int bt = blockIdx.x;
    if (act && !act[bt]) return;
    int *c = cnt + (size_t)bt * nch;
    if (threadIdx.x == 0) { int run = 0; for (int k = 0; k < nch; k++) { int v = c[k]; c[k] = run; run += v; } tot[bt] = run; }
}
__global__ void fl_write_k(const float *det_, size_t N, const int *off, int nch, int *list_, const int *act, int rx, int ry)
{
    const int bt = blockIdx.y;
    if (act && !act[bt]) return;
    const float *det = det_ + (size_t)bt * N;
    int *list = list_ + (size_t)bt * N;
    size_t base = (size_t)blockIdx.x * FCH;
    // thread t owns the contiguous voxels [t*8, t*8+8) of the chunk
    constexpr int PER = FCH / 256;
    int flags = 0, c = 0;
    for (int k = 0; k < PER; k++) { size_t i = base + threadIdx.x * PER + k; if (i < N && (double)det[i] <= 0.0) { flags |= 1 << k; c++; } }
    __shared__ int sc[256];
    sc[threadIdx.x] = c;
    __syncthreads();
    for (int o = 1; o < 256; o <<= 1) {   // inclusive scan (Hillis-Steele)
        int v = (threadIdx.x >= (unsigned)o) ? sc[threadIdx.x - o] : 0;
        __syncthreads();
        sc[threadIdx.x] += v;
        __syncthreads();
    }
    int pos = off[(size_t)bt * nch + blockIdx.x] + sc[threadIdx.x] - c;
    for (int k = 0; k < PER; k++)
        if (flags & (1 << k)) {
            int i = (int)(base + threadIdx.x * PER + k);
            int px = i % rx, py = (i / rx) % ry, pz = i / (rx * ry);
            list[pos++] = (pz << 20) | (py << 10) | px;
        }
}
void voxjac_correct(VoxJacCtx &c, float *cp, cudaStream_t st)
{
    CK(cudaMemsetAsync(c.rowFold, 0, (size_t)c.rn[1] * c.rn[2], st));
    foldrows_k<<<cdiv(c.N, 256), 256, 0, st>>>(c.det, c.rn[0], c.rn[1], c.rn[2], c.rowFold, nullptr);
    voxfold_k<<<cdiv(c.ga.nn, 128), 128, 0, st>>>(c.ga, c.rn[0], c.rn[1], c.rn[2], c.gvs[0], c.gvs[1], c.gvs[2], c.tab[0], c.tab[1],
                                                  c.tab[2], c.det, c.J, cp, c.rowFold, c.tab2[0], c.tab2[1], c.tab2[2], 0, nullptr, nullptr, 0, 0, 0, 0, nullptr, nullptr);
}

// ---- batched end-of-level correction: all templates of a chunk in parallel, device resident control flow (same arithmetic and
// pass structure as the sequential loop: evaluate; if NaN correct; at most 20 corrections; NaN after the 20th -> restore best DOF)
__global__ void vj_decide_k(const double *partial, int nblk, int *act)
{
    int b = blockIdx.x;
    if (!act[b]) return;
    __shared__ double sh[32];
    double s = 0.0;
    for (int i = threadIdx.x; i < nblk; i += blockDim.x) s += partial[(size_t)b * nblk + i];
    s = block_sum(s, sh);
    if (threadIdx.x == 0) act[b] = (s != s) ? 1 : 0;
}
__global__ void vj_count_k(int *act, int *iters, int nt)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nt && act[b]) iters[b]++;
}
__global__ void vj_restore_k(float *cp, const float *best, size_t ndof, const int *act)
{
    int b = blockIdx.y;
    if (!act[b]) return;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ndof) cp[(size_t)b * ndof + i] = best[(size_t)b * ndof + i];
}
void voxjac_fix_batch(VoxJacCtx &c, float *cp, const float *best, int nt, cudaStream_t st, int *nfold, int *nfail)
{
    if (c.rn[0] >= 1024 || c.rn[1] >= 1024 || c.rn[2] >= 1024) { fprintf(stderr, "voxjac_fix_batch: image dimensions >= 1024 are not supported\n"); exit(1); }
    const size_t ndof = 3 * c.ga.nn;
    int *dact = c.dact;
    CK(cudaMemsetAsync(c.diters, 0, nt * sizeof(int), st));
    if (c.dirty) CK(cudaMemsetAsync(c.dirty, 1, (size_t)nt * c.ncell, st));   // first evaluation: every cell
    {
        std::vector<int> one(nt, 1);
        CK(cudaMemcpyAsync(dact, one.data(), nt * sizeof(int), cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));
    }
    for (int pass = 0; pass < 20; pass++) {
        voxjac_k<<<dim3(c.nblk, nt), 256, 0, st>>>(c.ga, c.rn[0], c.rn[1], c.rn[2], c.tab[0], c.tab[1], c.tab[2], cp, c.det, c.J, c.partial, ndof, dact,
                                                  c.dirty, c.cn[0], c.cn[1], c.ncell);
        if (c.dirty) CK(cudaMemsetAsync(c.dirty, 0, (size_t)nt * c.ncell, st));
        vj_decide_k<<<nt, 256, 0, st>>>(c.partial, c.nblk, dact);
        vj_count_k<<<cdiv(nt, 32), 32, 0, st>>>(dact, c.diters, nt);
        const int nch = (int)((c.N + FCH - 1) / FCH);
        fl_count_k<<<dim3(nch, nt), 256, 0, st>>>(c.det, c.N, c.fcnt, nch, dact);
        fl_scan_k<<<nt, 32, 0, st>>>(c.fcnt, nch, c.ftot, dact);
        fl_write_k<<<dim3(nch, nt), 256, 0, st>>>(c.det, c.N, c.fcnt, nch, c.flist, dact, c.rn[0], c.rn[1]);
        voxfold_k<<<dim3(cdiv(c.ga.nn, 128), nt), 128, 0, st>>>(c.ga, c.rn[0], c.rn[1], c.rn[2], c.gvs[0], c.gvs[1], c.gvs[2], c.tab[0], c.tab[1],
                                                                c.tab[2], c.det, c.J, cp, nullptr, c.tab2[0], c.tab2[1], c.tab2[2], ndof, dact, c.dirty, c.cn[0], c.cn[1], c.cn[2], c.ncell,
                                                                c.flist, c.ftot);
        if (pass % 4 == 3 || pass == 0) {
            // early exit when no template is left (one tiny host read every few passes)
            std::vector<int> h(nt);
            CK(cudaMemcpyAsync(h.data(), dact, nt * sizeof(int), cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            bool any = false;
            for (int v : h) any |= (v != 0);
            if (!any) break;
        }
    }
    vj_restore_k<<<dim3(cdiv(ndof, 256), nt), 256, 0, st>>>(cp, best, ndof, dact);
    std::vector<int> h(nt), it(nt);
    CK(cudaMemcpyAsync(h.data(), dact, nt * sizeof(int), cudaMemcpyDeviceToHost, st));
    CK(cudaMemcpyAsync(it.data(), c.diters, nt * sizeof(int), cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    *nfold = 0; *nfail = 0;
    for (int b = 0; b < nt; b++) { *nfold += (it[b] > 0); *nfail += (h[b] != 0); }
}

}  // namespace f3d
