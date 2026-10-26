// SPDX-License-Identifier: BSD-3-Clause
// internal.cuh - shared declarations for the segem implementation (not part of the public API)
#pragma once
#include "segem.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#define CUDA_CHECK(x)                                                                                  \
    do {                                                                                               \
        cudaError_t e_ = (x);                                                                          \
        if (e_ != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
            abort();                                                                                   \
        }                                                                                              \
    } while (0)

namespace segem {

constexpr int kTileX = 128;          // x-extent of a "tile" (thread block of the per-voxel kernels)
constexpr int kMaxBasis = 35;        // basis functions of a 3D order-4 polynomial
constexpr int kMaxOrder = 4;
constexpr int kMaxMoments = 165;     // monomials x^p y^q z^r with p+q+r <= 2*kMaxOrder
constexpr int kEmGrid = 1024;         // blocks of the grid-stride reductions
constexpr int kLinesPerBlock = 16;    // bias-field lines per block of the moment kernel
constexpr int kSpecGrid = 64;        // blocks of the border-voxel kernels

// Device state of the EM (one instance in device memory)
struct EMState {
    float M[kMaxClasses];
    float V[kMaxClasses];
    float inv_v[kMaxClasses];
    float inv_sqrt[kMaxClasses];
    double loglik, oldloglik;
    int iter;
    int biasActive;          // set once a bias field has been fitted (coef != 0)
    int biasRan;             // bias fit executed in the current iteration
    int numMasked;
    int firstMasked;         // linear index of the first voxel in the mask (seg_EM quirk: out-of-mask MRF neighbours alias to it)
    float rmin, rmax;        // intensity range inside the mask (NiftySeg rescale_min/max)
    float coef[kMaxBasis];   // bias field polynomial coefficients
    double dummy;
};

struct Dims {
    int nx, ny, nz;
    size_t N;
    int ntilesX;             // ceil(nx / kTileX)
};

struct EmContext::Impl {
    Dims d;
    int C = 0;                       // number of classes
    EmParams prm;
    // inputs of the current run (not owned)
    const float *prior = nullptr;    // [C][N]
    const uint8_t *emMask = nullptr; // [N]
    const float *targetPtr = nullptr;
    // buffers
    float *expec = nullptr;          // [C][N]  (also the final posterior)
    float *expec2 = nullptr;         // [C][N]  second posterior buffer (ping-pong; the MRF field is evaluated on the fly)
    float *eFinal = nullptr;         // posterior of the last iteration (expec or expec2)
    float *T = nullptr;              // [N]     normalised log image
    float *bc = nullptr;             // [N]     bias corrected
    uint8_t *maskS[4] = {nullptr, nullptr, nullptr, nullptr};   // morphology stages
    int32_t *wsI0 = nullptr;         // [N] int work (labels / union-find parent)
    int32_t *wsI1 = nullptr;         // [N] int work
    uint8_t *wsB0 = nullptr;         // [N] byte work
    uint8_t *wsB1 = nullptr;         // [N] byte work
    uint8_t *wsB2 = nullptr;         // [N] byte work
    float *wsF0 = nullptr;           // [N] float work (smooth mask)
    float *wsF1 = nullptr;           // [N] float work
    int *voxList = nullptr;          // [N] sorted linear indices of the masked voxels
    float *pmin = nullptr, *pmax = nullptr;
    int *numActive = nullptr;        // device scalar
    void *cubTmp = nullptr; size_t cubTmpBytes = 0;
    EMState *st = nullptr;           // device
    EMState *hostSt = nullptr;       // pinned mirror
    double *red = nullptr;           // reduction partials (kEmGrid * 3*kMaxClasses doubles)
    double *lineAcc = nullptr;       // bias line partials
    double *momAcc = nullptr;        // bias moment partials
    size_t bytes = 0;
    StageTimes tm;
    cudaEvent_t ev[10];
    cudaStream_t stream = 0;
    int iters = 0;
    int nActHost = 0;
    int nSpecHost = 0;
    unsigned liveMask = 0;           // classes with a non-zero prior (class 0 and every class that has prior mass)
    int hasDormant = 0;              // some class has an identically zero prior
    int *specList = nullptr;         // [boundary voxels] masked voxels on the volume border (sorted)
    int *numSpec = nullptr;
    uint8_t *dormFlag = nullptr;     // [N] voxel fell back to the uniform posterior (dormant planes non-zero)
    double *redS = nullptr, *redSL = nullptr, *pT = nullptr;
    void *cubTmp2 = nullptr; size_t cubTmpBytes2 = 0;

    void allocate();
    void freeAll();
    cudaStream_t allocStream = 0;    // stream-ordered pool allocation (see EmContext ctor): run() must use this stream
    template <class T> T *dalloc(size_t n) { T *p; CUDA_CHECK(cudaMallocAsync(&p, n * sizeof(T), allocStream)); bytes += n * sizeof(T); return p; }
};

// ---- morph.cu
void launch_morphology(EmContext::Impl &I, const uint8_t *mask0, int dilate_radius, int erode_radius);
void launch_dilate_erode_u8(const uint8_t *in, uint8_t *out, uint8_t *tmp, const Dims &d, int radius, bool dilate, cudaStream_t s);
void launch_smooth_mask(EmContext::Impl &I);   // GetBiasCorrected mask: dilate 7, erode 3, gauss sigma 3 voxels -> wsF0
// ---- em.cu
void launch_em(EmContext::Impl &I, const float *target);
void launch_bias_corrected(EmContext::Impl &I);

} // namespace segem
