// SPDX-License-Identifier: BSD-3-Clause
// GPU (CUDA) batched symmetric block-matching affine registration: a port of NiftyReg's reg_aladin_sym
// (block matching, LTS regression, symmetric update).
//
// All images are device-resident float volumes, x fastest, NiftyReg sform conventions.
// Result matrices map reference (target) world coordinates -> floating (template) world coordinates,
// the convention of reg_aladin's output affine.
#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "mat4.h"

namespace aladin {

struct Params {
  // Defaults are the defaults of reg_aladin (NiftyReg reg-lib/_reg_aladin.cpp).
  int num_levels = 3;          // NumberOfLevels
  int levels_to_perform = 3;   // LevelsToPerform (reg_aladin -lp)
  int max_iter = 5;            // MaxIterations (x2 on first level, rigid x4 more on first level)
  bool rigid = true;           // PerformRigid
  bool affine = true;          // PerformAffine
  int block_percentage = 50;   // BlockPercentage
  int inlier_lts = 50;         // InlierLts
  int block_step = 1;          // BlockStepSize (1 or 2 supported)
  bool align_centre = true;    // AlignCentre
  bool fast_solver = false;    // true: single precision LTS solver on all levels (~15% faster; trajectory no longer follows the CPU's)
  bool exact = false;          // replicate the CPU float/double rounding sequence (pyramid, warping); see README, "Exact mode"
  float smoothing_sigma_voxel = 0.7355f;  // pyramid down-sampling gaussian (voxels), as reg_downsampleImage
};

struct Image {
  const float* dev = nullptr;  // device pointer, nx*ny*nz floats, x fastest
  int nx = 0, ny = 0, nz = 0;
  Mat44 sform;                 // voxel -> world (sto_xyz)
};

struct Timings {  // all in ms, measured with cudaEvents on the work stream
  float target_setup = 0;      // target pyramid + block statistics (once per target)
  float template_setup = 0;    // template pyramids + block statistics (once per template set)
  float level[8] = {0};        // registration time per level (includes all iterations)
  int levels = 0;
  float registration = 0;      // sum of levels (+ init)
  float total() const { return target_setup + template_setup + registration; }
};

class TemplateSet;

class AladinBatch {
 public:
  // Builds the target pyramid and the block statistics (done once per target).
  AladinBatch(const Image& target, const Params& p = Params(), cudaStream_t stream = 0);
  ~AladinBatch();
  AladinBatch(const AladinBatch&) = delete;
  AladinBatch& operator=(const AladinBatch&) = delete;

  // Registers all templates of the set (blocking on the stream). out_host: n matrices.
  // init (optional, n matrices): initial transformation (reg_aladin -inaff); skips align-centre.
  // out_dev (optional): device copy of the n result matrices (row-major float[16] each).
  void run(const TemplateSet& tset, Mat44* out_host, const Mat44* init = nullptr, float* out_dev = nullptr);

  // Convenience: builds the TemplateSet on the fly.
  void run(const Image* templates, int n, Mat44* out_host, const Mat44* init = nullptr);

  const Timings& timings() const { return tm_; }
  // Non-zero when a registration failed for some template (too few block correspondences).
  const std::vector<int>& status() const { return status_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Timings tm_;
  std::vector<int> status_;
};

// Pyramids and block selection of a set of templates (independent of the target, reusable
// across targets).
class TemplateSet {
 public:
  TemplateSet(const Image* templates, int n, const Params& p = Params(), cudaStream_t stream = 0);
  ~TemplateSet();
  TemplateSet(const TemplateSet&) = delete;
  TemplateSet& operator=(const TemplateSet&) = delete;
  int size() const;
  float setup_ms() const;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aladin
