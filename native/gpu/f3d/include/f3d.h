// SPDX-License-Identifier: BSD-3-Clause
// GPU implementation of NiftyReg F3D (reg_f3d<float>, LNCC, cubic B-spline, conjugate gradient) batched over templates
// registered to one target.  All image inputs and the control point outputs are device resident.
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "f3d_geom.h"
#include "f3d_state.h"

namespace f3d {

struct F3DParams {
    // Defaults are the defaults of NiftyReg reg_f3d (reg-lib/_reg_f3d.cpp, _reg_base.cpp).
    float beWeight = 0.001f;        // SetBendingEnergyWeight
    float leWeight = 0.01f;         // SetLinearEnergyWeight
    float jlWeight = 0.0f;          // SetJacobianLogWeight
    int levelNumber = 3;            // SetLevelNumber
    int levelToPerform = 3;         // SetLevelToPerform (number of levels actually run, coarsest first)
    int maxIterations = 150;        // SetMaximalIterationNumber (scaled by 2^(levelToPerform-1) for the first level)
    float lnccSigmaMM = 5.0f;       // UseLNCC(tp, sigma): gaussian std in mm
    float spacingVox = -5.0f;       // SetSpacing(0, s): negative = voxels of the full resolution reference
    bool verbose = false;
    // performance/accuracy trade-off: bit l set -> pyramid level l (0 = coarsest performed level) accumulates the LNCC Gaussian
    // convolutions in plain float instead of emulating NiftyReg's double accumulation of float products (faster, not CPU-faithful)
    int fastLevelMask = 0;
    // numerics = 1 ("fast"): same algorithm, different arithmetic. Levels in fastLevelMask (default: 6 = levels 1 and 2; level 0 stays exact)
    // use plain float LNCC kernels, the brick warp, ... (see README, "Numerics modes"). 0 = exact mode (bit identical to the unoptimised code).
    int numerics = 0;
    // execution only (no effect on results): the batch is split into this many groups of templates, each driven by its own CUDA stream
    int groups = 8;
};

struct F3DResult {
    // per template, per performed level
    int iterations[8] = {0};        // optimiser iteration counter at the end of the level (NiftyReg's counter)
    int gradEvals[8] = {0};
    int valueEvals[8] = {0};
    double finalObjective[8] = {0}; // best objective at the end of each level
    double finalLncc[8] = {0};      // weighted similarity part
};
struct F3DTiming {
    double setupMs = 0;             // pyramids, grids, allocation
    double levelMs[8] = {0};        // per level (all templates)
    double totalMs = 0;
    int macroSteps[8] = {0};
    double correctMs[8] = {0};      // end-of-level voxel based folding correction (CorrectTransformation)
};

class F3DBatch {
public:
    // d_target: device float image (nx*ny*nz, x fastest) with geometry tg; d_mask: optional device uint8 (nonzero = active)
    F3DBatch(const float *d_target, const ImgGeom &tg, const unsigned char *d_mask, const F3DParams &p, int maxBatch,
             cudaStream_t st = 0);
    ~F3DBatch();
    // Register n templates (device float images, all with geometry fg) to the target. aff[i] is the affine (reference->floating
    // world) the control point grid is initialised with. d_out[i] receives the final control point positions
    // (planar x,y,z in mm, 3*finalGrid().nnodes() floats, device).
    void run(const float *const *d_tpl, const ImgGeom &fg, int n, const M44 *aff, float *const *d_out, F3DResult *res,
             F3DTiming *tm);
    GridGeom finalGrid() const;
    // optimiser trace (one record per NiftyReg iteration print) of the last run: level, template
    const std::vector<LogRec> &trace(int level, int tpl) const;
    // per-evaluation trace (only when env F3D_TRACE_EVALS is set)
    const std::vector<EvalRec> &evtrace(int level, int tpl) const;
    struct Impl;
private:
    Impl *p_;
};

}  // namespace f3d
