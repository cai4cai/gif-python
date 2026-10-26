// SPDX-License-Identifier: BSD-3-Clause
// test_em.cu - compares the GPU EM (segem::EmContext) with the NiftySeg seg_EM class on synthetic volumes:
// nested ellipsoids with class dependent mean intensity, Gaussian noise and a smooth multiplicative bias field; the prior is a
// noisy one-hot map; optionally one class has an all-zero prior.  Reports the iteration counts, the agreement of the
// posteriors and of the hard labels, and of the bias corrected image.
//   usage: test_em [nx ny nz] [classes] [dormant 0|1] [mrf_beta] [bias_order] [seed]
#include "segem.h"
#include <dirent.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include "nifti1_io.h"
#include "_seg_common.h"
#include "_seg_tools.h"
#include "_seg_EM.h"
using namespace segem;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

static nifti_image *make_img(int nx, int ny, int nz, int nt) {
    int dims[8] = {3, nx, ny, nz, 1, 1, 1, 1};
    nifti_image *n = nifti_make_new_nim(dims, NIFTI_TYPE_FLOAT32, 0);
    n->dx = n->dy = n->dz = 1.f;
    n->dim[0] = n->ndim = nt > 1 ? 4 : 3; n->dim[4] = n->nt = nt; n->dim[5] = n->nu = 0;
    nifti_update_dims_from_array(n);
    n->datatype = NIFTI_TYPE_FLOAT32; n->nbyper = 4;
    n->data = calloc(n->nvox, 4);
    return n;
}

int main(int argc, char **argv) {
    int nx = argc > 3 ? atoi(argv[1]) : 50, ny = argc > 3 ? atoi(argv[2]) : 58, nz = argc > 3 ? atoi(argv[3]) : 42;
    int C = argc > 4 ? atoi(argv[4]) : 5;
    int dormant = argc > 5 ? atoi(argv[5]) : 1;
    float beta = argc > 6 ? (float)atof(argv[6]) : 0.15f;
    int order = argc > 7 ? atoi(argv[7]) : 3;
    unsigned seed = argc > 8 ? atoi(argv[8]) : 1;
    size_t N = (size_t)nx * ny * nz;
    std::mt19937 rng(seed);
    std::normal_distribution<float> G(0.f, 1.f);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    const int dormantClass = (dormant && C > 3) ? 3 : -1;
    // true classes: nested ellipsoids; class index assigned from the normalised radius, skipping the dormant class
    std::vector<int> cls(N), order_c;
    for (int c = 0; c < C; c++) if (c != dormantClass) order_c.push_back(c);
    std::vector<float> img(N), prior((size_t)C * N, 0.f);
    std::vector<uint8_t> mask(N);
    float bx = 0.15f * U(rng), by = 0.15f * U(rng), bz = 0.15f * U(rng), bxx = 0.1f * U(rng);
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        size_t v = x + (size_t)nx * (y + (size_t)ny * z);
        float px = (x - 0.5f * nx) / (0.5f * nx), py = (y - 0.5f * ny) / (0.5f * ny), pz = (z - 0.5f * nz) / (0.5f * nz);
        float r = sqrtf(px * px * 0.9f + py * py * 1.1f + pz * pz) * 1.05f;   // > 1 outside; the volume border is partly inside
        int k = r > 1.f ? 0 : 1 + (int)(r * (order_c.size() - 1));
        if (k >= (int)order_c.size()) k = (int)order_c.size() - 1;
        int c = order_c[k];
        cls[v] = c;
        float mean = 150.f + 220.f * k;
        float bias = 1.f + bx * px + by * py + bz * pz + bxx * px * px;
        img[v] = std::max(1.f, (mean + 25.f * G(rng)) * bias);
        mask[v] = r < 0.97f || (x + y + z) % 11 == 0 ? (r < 1.1f) : 0;
        // prior: mostly the true class, some mass on the adjacent class, 5% chance of a wrong confident class
        int c2 = order_c[std::min<int>(order_c.size() - 1, std::max(0, k + (U(rng) < 0.5f ? -1 : 1)))];
        float w1 = 0.65f + 0.25f * U(rng);
        if (U(rng) < 0.05f) { c2 = c; c = order_c[(k + 1) % order_c.size()]; }
        prior[(size_t)c * N + v] += w1;
        prior[(size_t)c2 * N + v] += 1.f - w1;
    }
    // ---------------- NiftySeg
    nifti_image *Tn = make_img(nx, ny, nz, 1), *Mn = make_img(nx, ny, nz, 1), *Pn = make_img(nx, ny, nz, C);
    memcpy(Tn->data, img.data(), N * 4);
    for (size_t i = 0; i < N; i++) ((float *)Mn->data)[i] = mask[i];
    memcpy(Pn->data, prior.data(), N * C * 4);
    seg_EM SEG(C, 1, 1);
    SEG.SetInputImage(Tn);
    SEG.SetMaskImage(Mn);
    SEG.SetPriorImage(Pn);
    SEG.SetVerbose(0);
    SEG.SetMaximalIterationNumber(30);
    SEG.SetMinIterationNumber(3);
    SEG.SetBiasField(order, 0.1f);
    SEG.SetMRF(beta);
    SEG.Run_EM();
    nifti_image *cres = SEG.GetResult();
    nifti_image *cbc = SEG.GetBiasCorrected((char *)"bc.nii.gz");
    const float *cr = (const float *)cres->data, *cb = (const float *)cbc->data;
    // ---------------- GPU
    float *dT, *dP; uint8_t *dM;
    CK(cudaMalloc(&dT, N * 4)); CK(cudaMalloc(&dP, N * C * 4)); CK(cudaMalloc(&dM, N));
    CK(cudaMemcpy(dT, img.data(), N * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dP, prior.data(), N * C * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dM, mask.data(), N, cudaMemcpyHostToDevice));
    unsigned live = EmContext::liveClasses(dP, C, N);
    EmParams p;
    p.max_iter = 30; p.min_iter = 3; p.bias_order = order; p.bias_ratio = 0.1f; p.mrf_beta = beta;
    EmContext ctx(nx, ny, nz, C, p, live);
    ctx.run(dP, dM, dT);
    std::vector<float> gs((size_t)C * N), gb(N);
    CK(cudaMemcpy(gs.data(), ctx.seg(), gs.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gb.data(), ctx.biasCorrected(), N * 4, cudaMemcpyDeviceToHost));
    // ---------------- compare
    size_t nm = 0, argDiff = 0; double maxd = 0, sumd = 0; double bmax = 0, bsum = 0; size_t nb = 0; double rngI = 0;
    for (size_t v = 0; v < N; v++) rngI = std::max(rngI, (double)img[v]);
    for (size_t v = 0; v < N; v++) {
        if (!mask[v]) continue;
        nm++;
        int ac = 0, ag = 0; float vc = -1, vg = -1;
        for (int c = 0; c < C; c++) {
            float a = cr[(size_t)c * N + v], b = gs[(size_t)c * N + v];
            double d = fabs((double)a - b); maxd = std::max(maxd, d); sumd += d;
            if (a > vc) { vc = a; ac = c; }
            if (b > vg) { vg = b; ag = c; }
        }
        if (ac != ag) argDiff++;
    }
    for (size_t v = 0; v < N; v++) { double d = fabs((double)cb[v] - gb[v]); if (cb[v] == cb[v]) { bmax = std::max(bmax, d); bsum += d; nb++; } }
    printf("dims %dx%dx%d classes %d dormant %d beta %.2f order %d live mask 0x%x masked voxels %zu\n", nx, ny, nz, C, dormant, beta, order, live, nm);
    printf("iterations gpu %d\n", ctx.times().em_iterations);
    printf("posterior |cpu-gpu|: max %.3g mean %.3g; argmax differs at %zu of %zu masked voxels (%.4f%%)\n", maxd, sumd / (nm * C), argDiff, nm, 100.0 * argDiff / nm);
    printf("bias corrected |cpu-gpu|: max %.3g (image max %.0f) mean %.3g\n", bmax, rngI, bsum / nb);
    bool ok = argDiff <= nm / 200 && bmax < 0.01 * rngI;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
