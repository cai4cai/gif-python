// SPDX-License-Identifier: BSD-3-Clause
// compare the device 3x3 helpers (nifti_mat33_* ports) with the real NiftyReg C functions, bit for bit
#include "test_common.h"
extern "C" {
#include "nifti1_io.h"
}
int main(int argc, char **argv)
{
    int n = 200000;
    std::mt19937 rng(1);
    std::vector<float> A(9 * n);
    for (int i = 0; i < n; i++) {
        // LE-like matrices: scaled/rotated identity plus noise of various magnitudes
        float sc = 0.5f + (rng() % 1000) / 1000.f * 1.5f, noise = powf(10.f, -(float)(rng() % 6));
        std::normal_distribution<float> nd(0.f, 1.f);
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) A[9 * i + r * 3 + c] = (r == c ? sc : 0.f) + noise * nd(rng);
    }
    float *dA = h2d(A.data(), A.size()), *dout = dalloc<float>(38 * n);
    m3_test(dA, dout, n);
    CK(cudaDeviceSynchronize());
    auto o = d2h(dout, 38 * n);
    size_t bad[5] = {0}; double mx[5] = {0};
    for (int i = 0; i < n; i++) {
        mat33 M; memcpy(&M, &A[9 * i], 36);
        float d = nifti_mat33_determ(M);
        mat33 Inv = nifti_mat33_inverse(M), P = nifti_mat33_polar(M), MM = nifti_mat33_mul(M, M);
        mat33 Ri = nifti_mat33_inverse(P), RM = nifti_mat33_mul(Ri, M);
        auto cmpv = [&](int k, const float *a, const float *b, int cnt) { for (int j = 0; j < cnt; j++) { if (a[j] != b[j] && !(a[j] != a[j] && b[j] != b[j])) { bad[k]++; mx[k] = fmax(mx[k], fabs(a[j] - b[j])); break; } } };
        cmpv(0, &o[38 * i + 0], &d, 1);
        cmpv(1, &o[38 * i + 1], &Inv.m[0][0], 9);
        cmpv(2, &o[38 * i + 10], &P.m[0][0], 9);
        cmpv(3, &o[38 * i + 19], &MM.m[0][0], 9);
        cmpv(4, &o[38 * i + 28], &RM.m[0][0], 9);
    }
    const char *nm[5] = {"determinant", "inverse", "polar", "mul(A,A)", "mul(inv(polar),A)"};
    for (int k = 0; k < 5; k++) printf("%-20s mismatching matrices: %zu of %d (max abs %g)\n", nm[k], bad[k], n, mx[k]);
    return 0;
}
