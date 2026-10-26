// SPDX-License-Identifier: BSD-3-Clause
// Internal API of the GPU F3D implementation (also used by the stage tests).
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "f3d_geom.h"
#include "f3d_util.h"

namespace f3d {

// ---------------------------------------------------------------- conv / pyramid
struct Taps {
    int radius = 0;
    std::vector<float> h;  // 2*radius+1 values (host)
    float *d = nullptr;    // device copy
};
// NiftyReg reg_tools_kernelConvolution kernel construction. sigma>0: mm, sigma<0: voxels.
// kind: 0=gaussian (radius=int(3*s)), 1=cubic spline (radius=int(2*s)). s = sigma/pixdim (float division, as in NiftyReg)
Taps make_taps(float sigma, float pixdim, int kind);
void free_taps(Taps &t);

// Planar multi-channel separable pass. buffers: [b][c][N] (batch stride bStride, channel stride N)
void conv_axis(const float *in, float *out, int C, int B, size_t bStride, const int dims[3], int axis,
               const Taps &t, cudaStream_t st, bool exact = true, const float *dnorm = nullptr, const int *act = nullptr,
               const unsigned *list = nullptr, int count = 0);   // list: only these voxel indices are computed (others untouched)
// 1-D density table for an axis (NiftyReg: sum of the kernel weights inside the image), float
std::vector<float> density_table(const Taps &t, int n);

// Gaussian smoothing sigma_vox=0.7355 then nearest-neighbour downsample (NiftyReg reg_downsampleImage, type 1)
// returns newly allocated device image (geometry gout)
float *pyramid_step(const float *d_in, const ImgGeom &gin, ImgGeom &gout, cudaStream_t st);

// Build the NiftyReg image pyramid. lv[levelToPerform] outputs (device buffers allocated here)
void build_pyramid(const float *d_img, const ImgGeom &g0, int levelNumber, int levelToPerform,
                   std::vector<float *> &lv, std::vector<ImgGeom> &gl, cudaStream_t st);

// reg_intensityRescale(image, 0, 0, 1) on device (in place)
void rescale01(float *d, size_t n, cudaStream_t st);


// ---------------------------------------------------------------- transform
struct AxisTab {
    int pre;      // first node index (-1: voxel not covered, deformation stays 0)
    float w[4];   // cubic B-spline basis values
};
// per-axis lookup tables of the cubic B-spline basis exactly as NiftyReg evaluates them
std::vector<AxisTab> build_tab_generic(int n, float gvs);
std::vector<AxisTab> build_tab_lut(int n, int nnodes);  // NiftyReg's "spacing==5" look-up-table path

// control point grid, host helpers (positions in mm; planar x,y,z)
std::vector<float> grid_init_affine(const GridGeom &g, const M44 &aff);
// refine node positions (device, in/out sizes given by geoms)
void grid_refine(const float *d_old, const GridGeom &gold, float *d_new, const GridGeom &gnew, int B, cudaStream_t st);

struct WarpArgs {
    int rn[3];        // reference (and deformation field) dims
    int gn[3];        // control point grid dims
    int fn[3];        // floating dims
    float fijk[12];   // floating sto_ijk rows 0..2 (3x4)
    const AxisTab *tab[3];
    const int *mask;  // reference mask (>-1 active), may be null
    int bx;           // row-mode block width (0: generic per-voxel kernel); set by warp_prepare
    // cells (voxels sharing the same 4x4x4 control points) lying completely outside the floating image are flagged before the warp and
    // their voxels skipped (NaN); cflag [B][ncell] bytes provided by the caller (nullptr: off)
    unsigned char *cflag;
    int cn[3];
    size_t ncell;
    // fast numerics: brick kernel (one 5x5x5 cell per block, separable evaluation of the B-spline in shared memory, plain float). Needs the
    // look-up-table structure (every axis: voxel i belongs to cell i/5 with the basis values of position i%5), checked by warp_prepare.
    bool lut5;
    bool fast;
    float lw[3][5][4];
};
// enable the shared-memory staged warp kernel when the node footprint of a block fits (tabs: the 3 host tables)
void warp_prepare(WarpArgs &a, const std::vector<AxisTab> *tabs);
// B-spline deformation field + resample (linear) [+ image gradient] [+ store deformation field]
void warp(const WarpArgs &a, int B, const float *cpp, size_t cppStride, const float *flo, size_t floStride,
          float *warped, float *wgrad, float *def, const int *act, cudaStream_t st);


// ---------------------------------------------------------------- LNCC
struct LnccCtx {
    int dims[3] = {0, 0, 0};
    size_t N = 0;
    int B = 0;
    Taps gt[3];                  // Gaussian taps per axis (LNCC kernel)
    const float *ref = nullptr;  // [N] rescaled reference image
    const int *mask = nullptr;   // nullable (>-1 active)
    float *bufA = nullptr, *bufB = nullptr, *bufC = nullptr;  // [B][6][N]
    double *partial = nullptr;   // [B][nblk][2]
    double *acc = nullptr;       // [B][2]: sum, count (value) ; [B][2..3]: gradient active count
    int nblk = 0, nblk_s = 0, nblk_used = 0, partStride = 0;
    bool exact = true, fused = true, gp_done = false;
    int lfver = 2;               // fast LNCC kernel generation (1: first version, 2: blocked passes)
    float *Px = nullptr, *Pxy = nullptr, *Pxyz = nullptr;   // [3][N] template independent smoothed reference channels (all voxels valid)
};
void reduce_partials(LnccCtx &c, int nblk, int accOff, const int *act, cudaStream_t st);
void finalize_value(LnccCtx &c, double *out, const int *act, cudaStream_t st);
bool lncc_stream_supported(const LnccCtx &c);
void lncc_value_fused(LnccCtx &c, const float *warped, double *out, const int *act, bool exact, cudaStream_t st);
bool lncc_gradient_smooth_fused(LnccCtx &c, const float *warped, const float *wgrad, float *voxgrad, const int *act, cudaStream_t st);
void lncc_stats_fused(LnccCtx &c, const float *warped, float *stats, const int *act, bool exact, cudaStream_t st);
void lncc_init(LnccCtx &c, const int dims[3], const float pixdim[3], float sigma_mm, const float *d_ref, const int *d_mask, int B, bool exact = true);
void lncc_free(LnccCtx &c);
void lncc_build_ref_tables(LnccCtx &c);
// fast numerics LNCC kernels (f3d_lncc_fast.cu); return false when not applicable (mask, unsupported radii)
bool lnccf_value(LnccCtx &c, const float *warped, const int *act, cudaStream_t st);
bool lnccf_stats(LnccCtx &c, const float *warped, const int *act, cudaStream_t st);
bool lnccf_smooth(LnccCtx &c, const float *warped, const float *wgrad, float *voxgrad, const int *act, cudaStream_t st);
// LNCC value for each template: out[b] = mean |local ncc| (double); NaN if no valid voxel
void lncc_value(LnccCtx &c, const float *warped, double *out, const int *act, cudaStream_t st);
// voxel based LNCC gradient (NiftyReg reg_getVoxelBasedLNCCGradient); voxgrad [B][3][N] is overwritten
void lncc_gradient(LnccCtx &c, const float *warped, const float *wgrad, float *voxgrad, const int *act, cudaStream_t st);


// ---------------------------------------------------------------- voxel gradient -> node gradient
struct GradCtx {
    int dims[3] = {0, 0, 0};
    size_t N = 0;
    int B = 0;
    Taps t[3];
    float *dens[3] = {nullptr, nullptr, nullptr};
    unsigned *lst[3] = {nullptr, nullptr, nullptr};   // pruned passes: voxels whose smoothed value is read by vox2node (and the dilations)
    int lcnt[3] = {0, 0, 0};
    bool exact = true;   // false: plain float accumulation (fast numerics)
};
// restrict the three smoothing passes to the voxels vox2node reads (identical values there, ~4x fewer convolution outputs)
void gradconv_prune(GradCtx &g, const ImgGeom &ref, const GridGeom &grid);
void gradconv_init(GradCtx &g, const int dims[3], const float pixdim[3], const float nodeSpacing[3], int B);
void gradconv_free(GradCtx &g);
// cubic spline smoothing of the 3 gradient components along x, y, z (separate NiftyReg calls). Returns pointer to result
// (either vg or tmp, both [B][3][N])
float *gradconv_apply(GradCtx &g, float *vg, float *tmp, const int *act, cudaStream_t st);
// reg_voxelCentric2NodeCentric (update=false): nodeGrad [B][3][nnodes]
void vox2node(const ImgGeom &ref, const GridGeom &grid, const M44 &floIJK, float weight, const float *vg, float *nodeGrad,
              int B, const int *act, cudaStream_t st);


// ---------------------------------------------------------------- regularisation (control point grid)
struct GridArgs {
    bool fast = false;     // fast numerics (float polar decomposition etc.)
    int n[3];
    size_t nn;
    float xyz[9], ijk[9];  // 3x3 parts of the grid sto_xyz / sto_ijk (row major)
    float lat[12];         // grid sto_xyz 3x4
    float d[3];            // grid spacing (mm)
    float rd[3];           // reference voxel size (mm)
};
struct PenaltyOut {
    double jac, be, le;    // weighted penalty values (jac may be NaN)
    int restored;          // 1 if the folding correction failed and the best DOF were restored
};
void set_fuse_masks(int le, int be, int be2);
void m3_test(const float *dA, float *dout, int n);
GridArgs make_grid_args(const GridGeom &g, const ImgGeom &ref);
// per template block: [cur = best + scale*grad] -> Jacobian penalty (+ folding correction loop) -> BE -> LE.
size_t penalty_scratch_floats(const GridGeom &g);
struct PenaltyBufs {
    double *partial = nullptr;
    int *fold = nullptr;
    int *fix = nullptr;
    int nblk = 0;
};
void penalty_bufs_init(PenaltyBufs &pb, const GridGeom &g, int B);
void penalty_bufs_free(PenaltyBufs &pb);
void penalty_value(const GridArgs &g, PenaltyBufs &pb, float *cur, const float *best, const float *grad, size_t cpStride,
                   float *scratch, size_t scrStride, const float *scale, float jlw, float bew, float lew, PenaltyOut *out, int B,
                   const int *act, cudaStream_t st);
// gradients (added to nodeGrad [B][3nn]); cp is the current grid (BE gradient round-trips cp through displacements like NiftyReg)
void be_gradient(const GridArgs &g, float *cp, float *nodeGrad, size_t cpStride, float *scratch, size_t scrStride, float weight,
                 int B, const int *act, cudaStream_t st);
void le_gradient(const GridArgs &g, const float *cp, float *nodeGrad, size_t cpStride, float *scratch, size_t scrStride, float weight,
                 int B, const int *act, cudaStream_t st);
void jac_gradient(const GridArgs &g, const float *cp, float *nodeGrad, size_t cpStride, float *scratch, size_t scrStride, float weight,
                  int B, const int *act, cudaStream_t st);


// ---------------------------------------------------------------- optimiser control (device kernels)
struct OptArgs {
    size_t ndof;        // 3 * nodes
    int B;
    float simWeight;    // similarity weight (float used for objective) -- double below
    double simWeightD;
};
struct TState;
struct LogRec;
struct EvalRec;
void ctl_init_level(TState *S, EvalRec *ev, const double *lncc, const PenaltyOut *pen, double simW, float maxStep, int maxit, int B, cudaStream_t st);
void ctl_begin(TState *S, LogRec *log, int *actGrad, int B, cudaStream_t st);
void ctl_cg(TState *S, LogRec *log, EvalRec *ev, float *dbg, float *grad, float *arr1, float *arr2, float *cur, const float *best, const int *actGrad,
            int *trialOK, float *scale, size_t ndof, int B, cudaStream_t st);
void ctl_mid(const PenaltyOut *pen, const int *trialOK, int *actWarp, int B, cudaStream_t st);
void ctl_after(TState *S, LogRec *log, EvalRec *ev, float *cur, float *best, const double *lncc, const PenaltyOut *pen, double simW,
               const int *trialOK, int *trialOKnext, float *scale, size_t ndof, int B, int *doneCount, cudaStream_t st);


// ---------------------------------------------------------------- voxel-level Jacobian (end-of-level CorrectTransformation)
struct VoxTab {
    int pre;
    float w[4], f[4];   // cubic B-spline basis and first derivative
};
std::vector<VoxTab> build_voxtab(int n, float gvs);
struct VoxTab2 {
    float v[4], f[4];   // NiftyReg get_BSplineBasisValue(basis, index, value, first) for index 0..3 (double formulas)
};
struct VoxJacCtx {
    GridArgs ga;
    int rn[3];
    size_t N = 0;
    float gvs[3];
    VoxTab *tab[3] = {nullptr, nullptr, nullptr};
    VoxTab2 *tab2[3] = {nullptr, nullptr, nullptr};
    float *det = nullptr, *J = nullptr;   // [N], [9N]  (single template work buffers)
    double *partial = nullptr;
    unsigned char *rowFold = nullptr;   // [rz*ry] rows containing a det<=0 voxel
    int nblk = 0, Bc = 1;
    int *dact = nullptr, *diters = nullptr;
    unsigned char *dirty = nullptr;   // [Bc][ncell]: cells with a changed control point since the last determinant evaluation
    int cn[3] = {0, 0, 0};
    size_t ncell = 0;
    int *flist = nullptr, *fcnt = nullptr, *ftot = nullptr;   // sorted folded voxel indices, chunk counts/offsets, totals
};
void voxjac_init(VoxJacCtx &c, const GridArgs &ga, const GridGeom &g, const ImgGeom &ref, int Bc = 1);
void voxjac_free(VoxJacCtx &c);
// penalty (mean log(det)^2 over all reference voxels, NaN if any det<=0) of template grid cp (device, 3*nn floats)
double voxjac_penalty(VoxJacCtx &c, const float *cp, cudaStream_t st);
// one folding correction pass using the dets/matrices of the last voxjac_penalty call
void voxjac_correct(VoxJacCtx &c, float *cp, cudaStream_t st);
// end-of-level CorrectTransformation for nt templates (cp [nt][3nn] contiguous, nt <= c.Bc), restores `best` for templates that fail
void voxjac_fix_batch(VoxJacCtx &c, float *cp, const float *best, int nt, cudaStream_t st, int *nfold, int *nfail);

}  // namespace f3d
