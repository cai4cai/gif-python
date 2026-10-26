# gpu/aladin - batched symmetric block-matching affine registration (CUDA)

A GPU port of the NiftyReg `reg_aladin_sym` algorithm (NiftyReg is BSD-3-Clause, see `NOTICE`), batched over many floating images
registered to one reference image.

## What is ported
| NiftyReg function | here |
|---|---|
| `reg_downsampleImage` (5-tap gaussian pyramid + 2x sub-sampling) | `conv_axis_kernel`, `conv_axis_exact_kernel` |
| `_reg_set_active_blocks` (block variance ranking) | block statistics kernels |
| `reg_affine_getDeformationField`, `reg_resampleImage` (linear, NaN padding) | `warp_kernel`, `warp_exact_kernel` |
| `block_matching_method` (4x4x4 blocks, 7^3 search, NCC) | `bm_kernel` |
| `optimize` / `optimize_affine3D` / `optimize_rigid3D` (LTS regression, SVD) | `lts_kernel` |
| symmetric update `reg_mat44_logm/expm/avg2`, `nifti_mat44_inverse` | `mat4.h` |
| iteration plan of `reg_aladin::Run` | host loop in `aladin.cu` |

## API (`aladin.h`)
`aladin::Params` (defaults = defaults of `reg_aladin`), `aladin::Image` (device float volume + sform), `AladinBatch(target, params)`
builds the reference pyramid; `TemplateSet(images, n, params)` builds the floating pyramids; `AladinBatch::run(set, out_host)`
returns n 4x4 matrices (reference world -> floating world, row major float).

Exact mode (`Params::exact`) reproduces NiftyReg's float/double rounding sequence of the pyramid and of the warping (slow on GPUs
with a low fp64 rate); `fast_solver` uses a single precision LTS solver on all levels.

## Build and test
```
export NIFTYREG_PREFIX=<install prefix of a NiftyReg build with static libraries>
make                 # build/aladin_cli
make tools-cpu       # maths_check, cpu_trace, stage_check (link NiftyReg)
build/maths_check    # mat4.h log / exp / avg2 / inverse against NiftyReg's Eigen based functions
python3 tools/prep_data.py <outdir> a.nii.gz b.nii ...      # raw .f32/.hdr inputs of aladin_cli
ALADIN_LP=2 build/aladin_cli <outdir> <reps> target.hdr tmpl1.hdr ...
build/cpu_trace target.nii template.nii [levels]            # NiftyReg reference run with per-iteration matrices
ALADIN_EXACT=1 ALADIN_DUMP=/tmp/dump ALADIN_DUMP_IT=50 ALADIN_LP=2 build/aladin_cli ...;  build/stage_check target template /tmp/dump 50 <0|1> 1
```
`stage_check` feeds the intermediates dumped by the GPU (warped image, block matching correspondences) to the NiftyReg functions and
reports the differences.
