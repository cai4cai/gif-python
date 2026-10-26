// SPDX-License-Identifier: BSD-3-Clause
// gif_stages.h - GPU stages of the gif-python multi-atlas pipeline that are specific to gif-python (written from gif-python's
// Python code; the registration and EM stages are the ports in ../gpu).
//
//   Resampler    reg_resample semantics (padding 0, linear / nearest neighbour, rounding to the integer type of the floating image)
//                of an atlas image and an atlas label map with a cubic B-spline control point grid.
//   LnccWeights  src/multi_atlas/utils.py get_lncc_distance + exp(-distance / temperature): scipy-style separable Gaussian
//                (mode 'reflect'), float32 intermediates, same operation order.
//   Fusion       src/multi_atlas/multi_atlas_fusion.py: label probabilities (kept sparse), argmax, tissue prior, final parcellation.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "f3d_internal.h"

namespace gifgpu {

constexpr int kMaxAtlases = 64;
constexpr int kMaxTissues = 12;

// float label map -> uint16, through float16 as gif-python reads label maps: get_fdata(dtype=float16).astype(uint16); NaN -> 0
void labels_from_float(const float *d_in, uint16_t *d_out, size_t n, cudaStream_t st);
// d_present[65536] (bytes): set to 1 for every label value that occurs
void label_presence(const uint16_t *d_lab, size_t n, unsigned char *d_present, cudaStream_t st);

class Resampler {
public:
    // ref: target image geometry, grid: control point grid geometry, flo: floating (atlas) geometry
    Resampler(const f3d::ImgGeom &ref, const f3d::GridGeom &grid, const f3d::ImgGeom &flo, cudaStream_t st);
    ~Resampler();
    Resampler(const Resampler &) = delete;
    Resampler &operator=(const Resampler &) = delete;
    // d_cpp: control point positions (planar x,y,z in mm).  Linear interpolation of d_flo_img into d_img_out (float, rounded as
    // reg_resample rounds for the NIfTI datatype `img_datatype` of the floating image) and/or nearest neighbour of d_flo_lab into
    // d_lab_out.  Outputs may be null.  d_flo_img must always be given (the deformation is computed together with the warp).
    void run(const float *d_cpp, const float *d_flo_img, int img_datatype, float *d_img_out, const uint16_t *d_flo_lab,
             uint16_t *d_lab_out);

private:
    struct Impl;
    Impl *p_;
};

class LnccWeights {
public:
    // sigma_cfg: config.lncc_sigma (negative: mm, positive: voxels); spacing: voxel size of the target (header zooms)
    LnccWeights(const float *d_target, const int dims[3], const float sigma_cfg[3], const float spacing[3], float temperature,
                cudaStream_t st);
    ~LnccWeights();
    LnccWeights(const LnccWeights &) = delete;
    LnccWeights &operator=(const LnccWeights &) = delete;
    // d_weights = exp(-(1 - LNCC) / temperature), NaN -> 0
    void run(const float *d_atlas_warped, float *d_weights);
    // true if the kernel needs more than the supported radius
    bool ok() const;
    const double *sigma() const;

private:
    struct Impl;
    Impl *p_;
};

class Fusion {
public:
    // labels: class index -> label (order of gif-python's labels_array); tissues[c]: tissues assigned to class c
    Fusion(size_t N, int dims[3], const std::vector<uint16_t> &labels, const std::vector<std::vector<int>> &tissues, int ntissue,
           cudaStream_t st);
    ~Fusion();
    Fusion(const Fusion &) = delete;
    Fusion &operator=(const Fusion &) = delete;
    void set_inputs(const std::vector<const uint16_t *> &lab, const std::vector<const float *> &w);
    // multi_atlas_proba_seg_argmax (uint16 labels)
    void argmax_labels(uint16_t *d_out);
    // multi_atlas_tissue_prior, class-major [ntissue][N]
    void tissue_prior(float *d_prior);
    // tissue segmentation from the EM posterior [ntissue][N] (float16 rounding, prior-background and border override, argmax)
    void tissue_seg(const float *d_posterior, const float *d_prior, uint8_t *d_tseg);
    // final parcellation (int32)
    void final_parcellation(const uint8_t *d_tseg, int32_t *d_out);

private:
    struct Impl;
    Impl *p_;
};

}  // namespace gifgpu
