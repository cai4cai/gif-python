# gpu/f3d - batched F3D (cubic B-spline, LNCC, conjugate gradient) registration (CUDA)

A GPU port of NiftyReg `reg_f3d<float>` with the LNCC similarity measure (NiftyReg is BSD-3-Clause, see `NOTICE`), batched over many
floating images registered to one reference image.  Control point grids are initialised from an affine matrix.

## What is ported
| NiftyReg | here |
|---|---|
| image pyramid, intensity rescale (`reg_createImagePyramid`, `reg_tools_rescaleImage`) | `src/f3d_conv.cu` |
| `reg_spline_getDeformationField`, `reg_resampleImage`, `reg_getImageGradient`, `reg_spline_refineControlPointGrid`, grid from affine | `src/f3d_transform.cu` |
| `reg_lncc` value and voxel gradient (gaussian window statistics) | `src/f3d_lncc.cu`, `f3d_lncc_stream.cu` (exact), `f3d_lncc_fast.cu` (fast) |
| `reg_voxelCentric2NodeCentric`, gradient smoothing | `src/f3d_gradient.cu` |
| `reg_spline_approxBendingEnergy*`, `reg_spline_approxLinearEnergy*`, `reg_spline_getJacobianPenalty*`, folding correction | `src/f3d_regul.cu` |
| `reg_optimiser` / `reg_conjugateGradient`, line search, `NormaliseGradient`, level loop of `reg_f3d` | `src/f3d_optim.cu`, `src/f3d_batch.cu` |

## API (`include/f3d.h`)
`F3DParams` (defaults = NiftyReg `reg_f3d` defaults; fields map to `SetBendingEnergyWeight`, `SetLinearEnergyWeight`,
`SetJacobianLogWeight`, `SetLevelNumber`, `SetLevelToPerform`, `SetMaximalIterationNumber`, `UseLNCC`, `SetSpacing`),
`F3DBatch(d_target, geom, d_mask, params, maxBatch)`, `run(templates, geom, n, affines, d_out, results, timing)`; `d_out[i]` receives
the final control point positions (planar x,y,z in mm, the content of NiftyReg's control point position image).

### Numerics modes
`numerics = 0` (exact): the LNCC convolutions emulate NiftyReg's double accumulation of float products and the penalty terms use
the float operation structure of the NiftyReg binaries; `numerics = 1` (fast) uses plain float accumulation on the levels in
`fastLevelMask`.  `groups` only changes how the batch is scheduled on streams.

## Build and test
```
export NIFTYREG_PREFIX=<install prefix of a NiftyReg build with static libraries>
make -j4 all                      # build/libf3d.a, the tests, build/f3d_run, build/ref_f3d
test/run_all_tests.sh target.nii.gz template.nii affine.txt     # stage tests against NiftyReg library functions
build/ref_f3d target template affine out_cpp.nii [verbose]       # NiftyReg reference run (settings of test/test_config.h)
build/test_stage6 target affine out_cpp.nii template             # full run against the NiftyReg grid
build/test_batch target affine reps 'dir/%s_cpp.nii' tpl1.nii ...
build/f3d_run target affine out_dir tpl1.nii ...                 # command line driver, writes NiftyReg format grids
```
The stage tests (`test_stage1/3/5`, `test_warpdf`, `test_m3`, `emul_*`, `test_fma_*`, `replica_be`) call the real NiftyReg functions
on the same inputs and report maximum differences.  `test/test_config.h` holds the (non-default) settings shared by the tests and by
`ref_f3d`; `F3D_TEST_BE` and `F3D_TEST_JL` override the two penalty weights on both sides.
Note: the optimisation is chaotic; after the first iterations of a level GPU and CPU trajectories separate (last-bit differences),
so end-to-end grids agree statistically, not bit for bit.
