// SPDX-License-Identifier: BSD-3-Clause
// segem.h - GPU port of the NiftySeg seg_EM expectation-maximisation segmentation (Gaussian mixture, MRF, polynomial bias
// field) and of the NiftySeg binary-mask morphology (Dillate, Erosion, Close_Forground_ConnectComp, GaussianFilter4D_cArray).
//
// Everything is device resident.  Single modality, up to kMaxClasses classes, bias field order 1..4.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace segem {

constexpr int kMaxClasses = 12;     // NiftySeg maxNumbClass

// Defaults are the defaults of the NiftySeg seg_EM application (seg-apps/seg_EM.cpp) and of the seg_EM class.
struct EmParams {
    float mrf_beta = 0.4f;        // SetMRF(beta); 0 disables the MRF term
    int max_iter = 100;           // SetMaximalIterationNumber
    int min_iter = 0;             // SetMinIterationNumber
    int bias_order = 3;           // SetBiasField(order, ratio): polynomial order, 1..4 supported
    float bias_ratio = 0.f;       // bias field is fitted when the relative likelihood improvement is below this (0: always)
    float conv_crit = 0.0005f;    // convergence criterion of the log-likelihood
};

struct StageTimes {                  // milliseconds, cudaEvent based
    float morph = 0, em_init = 0, em = 0, bias_corrected = 0, total = 0;
    int em_iterations = 0;
};

class EmContext {
public:
    // nx,ny,nz: image size; numClasses: 2..kMaxClasses (class 0 is the fallback class of seg_EM).
    // liveMask: bit c set = class c has a non-zero prior somewhere (see liveClasses); 0 = treat all classes as live.
    //   Classes without prior are handled as seg_EM does (they only receive posterior mass at volume border voxels and at
    //   voxels where the mixture likelihood underflows).
    // allocStream: buffers come from the stream-ordered device memory pool on this stream; run() must use the same stream.
    EmContext(int nx, int ny, int nz, int numClasses, const EmParams &p = EmParams(), unsigned liveMask = 0,
              cudaStream_t allocStream = 0);
    ~EmContext();
    EmContext(const EmContext &) = delete;
    EmContext &operator=(const EmContext &) = delete;

    // Classes (bit mask) that have a non-zero prior value at some voxel; bit 0 is always set.
    static unsigned liveClasses(const float *prior_dev, int numClasses, size_t numVoxels, cudaStream_t s = 0);

    // prior: device float [C][N] (class-major), mask: device uint8 [N] (non-zero = voxel is segmented),
    // target: device float [N] raw intensities.  Runs seg_EM and, if biasCorrected, GetBiasCorrected.
    void run(const float *prior, const uint8_t *mask, const float *target, cudaStream_t s = 0, bool biasCorrected = true);

    // Device-resident results of the last run
    const float *seg() const;            // [C][N]  posterior (seg_EM::GetResult)
    const float *biasCorrected() const;  // [N]     seg_EM::GetBiasCorrected
    const StageTimes &times() const;
    int numClasses() const;
    size_t numVoxels() const;
    size_t deviceBytes() const;

    // Binary mask morphology on a device uint8 [N] mask:  out1 = dilate(mask0, dilate_radius);  out2 = close_foreground(out1);
    // out3 = erode(out2, erode_radius).  Same results as NiftySeg Dillate / Close_Forground_ConnectComp / Erosion.
    // Stage buffers are available through maskStage(1..3) afterwards (maskStage(0) = copy of mask0).
    void morphology(const uint8_t *mask0, int dilate_radius, int erode_radius, cudaStream_t s = 0);
    const uint8_t *maskStage(int k) const;

    struct Impl;   // implementation detail (public only so the .cu files can name it)
private:
    Impl *p_;
};

}  // namespace segem
