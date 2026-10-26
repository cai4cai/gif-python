# native - GPU backend of gif-python

`gif_gpu` is a CUDA program that runs gif-python's multi-atlas pipeline (`src/multi_atlas/multi_atlas_segmentation.py`) on one GPU and
writes the same output files.  It is used through `GifConfig(backend="gpu")` or `gif-python segment --backend gpu`; the CPU backend
(NiftyReg / NiftySeg executables) is unchanged and remains the default.  Everything here is BSD-3-Clause (see `LICENSE`, `NOTICE`,
and `PROVENANCE.md` for the origin of every file).

## What is computed

The GPU backend reproduces what the CPU backend computes, with the same parameters (all taken from `GifConfig`):

| step (CPU backend) | parameters | GPU backend |
|---|---|---|
| `reg_aladin -ref target -flo atlas -lp 2` | symmetric (default of reg_aladin), rigid then affine, 3 levels of which `-lp 2` are run | `gpu/aladin` (port of `reg_aladin_sym`: block matching, LTS regression), batched over the atlases |
| `reg_f3d -aff affine -jl 0.0001 -be 0.005 -maxit 250 -ln 4 -lp 3 -sx -5 --lncc 5 --interp 1` | LNCC (Gaussian 5 mm), linear energy 0.01 (default), non-symmetric, cubic B-spline grid of 5 voxels refined from 10 voxels, conjugate gradient | `gpu/f3d` (port of `reg_f3d`), batched over the atlases; `numerics=0` emulates the arithmetic of NiftyReg, `numerics=1` is the validated fast option |
| `reg_resample -inter 0` of the atlas segmentation, `-inter 1` of the atlas image | padding 0, output type = type of the floating image (an int16 atlas image is rounded) | `Resampler` (`gif/gif_stages.cu`) with the B-spline field of the GPU F3D |
| LNCC distance and weights (`get_lncc_distance`) | Gaussian with sigma = 2.5 mm / voxel size per axis, radius `floor(3 sigma)`, scipy `reflect` boundary, float32; `weights = exp(-(1 - LNCC) / 0.15)`, NaN -> 0 | `LnccWeights` (same operation order) |
| fusion (`get_multi_atlas_proba_seg`), tissue prior (`get_multi_atlas_tissue_prior`) | label probability = sum of the weights of the atlases with that label; tissue prior = probabilities distributed over the tissues of the structure, 0.01 threshold, normalisation | `Fusion`: the probabilities are kept sparse (at most one entry per atlas and voxel), so no `H x W x D x labels` array exists |
| `seg_EM -priors4D prior -max_iter 30 -min_iter 3 -bc_order 4 -bc_thresh 0.05 -mrf_beta 0.1` | Gaussian mixture, MRF, polynomial bias field; no mask = all voxels | `gpu/seg` (port of NiftySeg `seg_EM`) |
| tissue segmentation, `get_structure_seg_from_tissue_seg` | argmax of the float16 posterior with the background / border override; best label whose tissues contain the tissue | `Fusion::tissue_seg`, `Fusion::final_parcellation` |

Output files (in `save_dir`, same names as the CPU backend): `multi_atlas_proba_seg_argmax.nii.gz` (uint16),
`multi_atlas_tissue_prior.nii.gz` (float32, 4D), `multi_atlas_tissue_seg.nii.gz` (float32 posterior, 4D),
`final_parcellation.nii.gz` (int32) and, with `gpu_write_warped_seg` (default), `registered_atlases/<name>/warped_atlas_seg.nii.gz`
(uint16 here, float32 in the CPU backend).  The affine, control point grid and warped images of the registration are not written
(`keep_registration_files` has no effect); `use_old_results` is ignored (everything is recomputed, it takes seconds).

## Build

Needs the CUDA toolkit (nvcc, tested with 13.0 on an NVIDIA GB10, sm_121), a C++17 compiler, OpenMP and zlib.

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release     # CMAKE_CUDA_ARCHITECTURES defaults to "native"
cmake --build native/build -j
native/build/gif_gpu --check-device                            # prints the GPU, exit status 5 without a CUDA device
```

The Python package finds `native/build/gif_gpu` of a source checkout; otherwise use `--gpu-binary`, `GifConfig(gpu_binary=...)`, the
environment variable `GIF_GPU_BINARY`, or put `gif_gpu` on `PATH`.  Importing gif-python never needs CUDA; a missing executable gives
a `BinaryNotFoundError` with these instructions and a missing device an `ExternalToolError`.

## Use

```bash
gif-python segment --backend gpu --img target.nii.gz --atlas-dir atlases/ --out out/        # same options as the CPU backend
```
```python
from src.utils.config import GifConfig
multi_atlas_segmentation(img, None, atlases, structure_csv, tissue_csv, out_dir, GifConfig(backend="gpu"))
```
`GifConfig` fields used by the GPU backend: all registration / weight / EM parameters of the CPU backend, plus `backend`, `gpu_binary`,
`gpu_write_warped_seg`, `gpu_batch` (atlases registered together, 0 = automatic from the image size so that the device memory stays
around 6 GB) and `gpu_numerics` (0 exact emulation of NiftyReg's arithmetic, 1 validated fast option, about 2x faster registration).
`multiprocessing`, `num_pools` and `omp` have no effect.  `gif_gpu --help` lists the options of the program (including the test modes
`--cpp-dir` and `--affine-dir`, which use the grids / affines of a CPU run, and `--dump-dir`).

## Limits (clear errors, nothing is silently ignored)

* No masks: a target mask or atlas masks (`-rmask`, `-fmask`) raise `NotImplementedError` (the registration ports do not implement
  masks; the mask of seg_EM and of the LNCC weights would be easy but is useless without the registered masks).
* `resample_method` must be 0 and `reg_f3d_interp` 1.
* One 3D image per file; the atlas images and segmentations must share size, geometry (sform or qform) and data type; at most 64
  atlases, 2 to 12 tissues, labels in 0..65535; LNCC kernel radius at most 40 voxels.
* Native byte order NIfTI-1 only (no NIfTI-2, no byte swapping).
* The results are not bit-identical to the CPU backend: the registration is a chaotic optimisation (see the validation).  Two ties are
  resolved differently: gif-python numbers the labels in the order of a Python `set`, the GPU backend in ascending order (only exact
  ties of `np.argmax` are affected); in the final parcellation, labels of probability zero (only reached when no label with positive
  probability has the right tissue) are tried in class order, numpy's `argpartition` returns another index among tied values.

## Tests

* `pytest -q` (repository root): the Python glue is tested with a stand-in executable (no GPU needed); the tests marked `gpu`
  (`tests/test_gpu_backend.py::test_gpu_end_to_end_on_a_phantom`, `tests/test_gpu_stages.py`) run when `gif_gpu` and a CUDA device are
  available and compare the weights, fusion, tissue prior and final parcellation with gif-python's own Python / numba functions on the
  same data (identity registration) and run the whole pipeline on a phantom.
* The ported NiftyReg / NiftySeg stages have comparison tests against the NiftyReg / NiftySeg libraries in `gpu/*/test*` (they need
  NiftyReg / NiftySeg static libraries; see the README of each directory), `bench/reg_scores.cpp` computes NCC / LNCC / NMI of warped
  atlases with NiftyReg's own measure classes.
* `scripts/gpu_validation/` contains the scripts of the validation below.

## Validation

Measured on `sie271-pc` (aarch64, NVIDIA GB10 sm_121, 20 cores, CUDA 13.0), NiftyReg / NiftySeg 1.5.77, Python 3.12.  Atlases: the 20
Mindboggle-101 OASIS-TRT-20 subjects (MNI space, 182x218x182, DKT31 + aseg labels, 113 labels including the background in the atlas segmentations,
6 tissue classes, `structures_info.csv` / `tissues_info.csv` of the package).  Targets: three IXI T1 volumes (256x256x150/146,
different orientation, 20 atlases each) and, for the leave-one-out, the T1 of the held-out atlas (19 atlases).  CPU backend = `gif-python
segment` with `--pools 2 --omp 8` (16 threads); GPU backend = default settings (`gpu_numerics=0`).  The scripts are in
`scripts/gpu_validation/`.

### Speed (20 atlases)

| target | CPU backend | GPU program (best of 3) | GPU `gif-python segment --backend gpu` (best of 3) | speed-up | `gpu_numerics=1` program / command |
|---|---|---|---|---|---|
| IXI002 | 233 s | 4.11 s | 5.08 s | 46x | 2.97 s / 4.01 s (58x) |
| IXI012 | 282 s | 5.37 s | 6.46 s | 44x | 3.38 s / 4.38 s (64x) |
| IXI035 | 246 s | 4.78 s | 5.80 s | 42x | 3.29 s / 4.32 s (57x) |

The CPU times are single runs on an otherwise quiet machine (the other CPU runs of the same targets, made while other work was running, took
242-295 s; the CPU backend is deterministic: a repeated run gave bit-identical outputs).  The GPU runs are on an idle GPU; the repeated
runs differ by less than 0.2 s.  "Command" includes the start of Python, the checks of the csv files and loading the result.
Stages of the GPU program (IXI002, seconds): read the 41 input files 0.25, upload and label check 0.07, `reg_aladin` 0.34,
`reg_f3d` 2.35 (3.58 / 2.97 for IXI012 / IXI035: depends on the number of iterations), resampling + LNCC weights of 20 atlases 0.51,
fusion + tissue prior 0.04, seg_EM 0.08, final parcellation 0.01, copy and compression of the 24 output files (overlapped) 0.2, process and
CUDA start 0.3.  CPU stages (IXI002): registration + weights of 20 atlases 199 s, the rest 33 s (tissue prior 6.5 s, `seg_EM` 5.6 s,
final parcellation 11.9 s, loading 5 s).  GPU memory in use by the program (sampled every 10 ms): 6.8-7.1 GB (`gpu_batch` automatic).

### Agreement with the CPU backend

The registration is a chaotic optimisation: a perturbation of 1e-7 of the starting affine changes the result.  The reference is therefore
the noise floor of the CPU backend itself (IXI002, `reg_aladin` result perturbed by +-1e-7 before `reg_f3d`):

| comparison (final parcellation) | voxels with a different label, % of the volume | % of the labelled voxels | mean Dice over 106 labels | tissue segmentation differs, % of the volume |
|---|---|---|---|---|
| **GPU vs CPU**, IXI002 | 1.43 | 10.6 | 0.893 | 1.12 |
| **GPU vs CPU**, IXI012 | 0.82 | 6.0 | 0.900 | 0.48 |
| **GPU vs CPU**, IXI035 | 0.85 | 6.5 | 0.907 | 0.48 |
| GPU (`gpu_numerics=1`) vs CPU, IXI002 / IXI012 / IXI035 | 1.39 / 0.86 / 0.84 | 10.3 / 6.3 / 6.4 | 0.904 / 0.897 / 0.908 | 1.09 / 0.51 / 0.47 |
| CPU vs CPU + 1e-7 (two perturbations), IXI002 | 1.60, 1.48 | 11.8, 10.9 | 0.892, 0.909 | 1.29, 1.22 |
| CPU + 1e-7 vs CPU + 1e-7 (the two), IXI002 | 0.76 | 5.7 | 0.918 | 0.48 |
| GPU vs GPU + 1e-7 (two perturbations and between them), IXI002 | 0.72, 0.74, 0.58 | 5.4, 5.5, 4.4 | 0.913, 0.917, 0.941 | 0.44, 0.47, 0.37 |
| GPU registration started from the affines of the CPU run (`--affine-dir`), IXI002 | 0.57 | 4.4 | 0.938 | 0.36 |

The GPU - CPU differences are of the size of the differences between two CPU runs with a 1e-7 different starting point.  IXI002 is
higher because the GPU registration of one atlas (OASIS-TRT-20-3) ended in a poor local optimum (global NCC of the warped atlas 0.27
against 0.66 on the CPU).  This is not specific to the GPU: `reg_f3d` of NiftyReg started from the GPU's affine for that atlas gives the
same poor result (NCC 0.27); other atlas/target pairs fail on the CPU too (IXI012, atlas 16: CPU NCC 0.52, GPU 0.65; IXI035, atlas 17:
CPU 0.66, GPU 0.57).

The affine registrations of the GPU and the CPU `reg_aladin` differ by 0.8 mm (median over the 60 atlas/target pairs of the mean displacement
over the brain voxels; 90th percentile 2.1 mm, maximum 3.4 mm; no systematic bias, standard deviation of the mean shift over the atlases
0.2-0.6 mm); `gif_gpu --aladin-exact 1` (replicates NiftyReg's rounding sequence, 0.5 s instead of 0.3 s) reduces this to 0.67 mm (median, IXI002)
and does not change the picture, so it is off by default.

The stages after the registration agree much more closely: with the control point grids of the CPU run (`--cpp-dir`, IXI002) the warped
atlas segmentations are identical in all 20 atlases (0 voxels of 196 million), the linearly resampled atlas images are identical (0 voxels),
the weights differ by at most 1.1e-4 (absolute; 0.28% relative where w > 1e-3), the argmax label map in 1 voxel, the tissue prior is bit-identical in 95.7% of the entries and differs by at most 0.01 (the 0.01 threshold) in the others, and the final parcellation in 0.029% of the volume (mean Dice 0.9972, minimum
0.975), almost all of it from the tissue segmentation (the posterior of the GPU `seg_EM` differs from NiftySeg's by 2.3e-4 on average);
where the tissue segmentations agree the final parcellations differ in 1 voxel.

### Leave-one-out against the expert labels

8 held-out subjects (OASIS-TRT-20-1, 3, 5, 7, 9, 11, 13, 15), 19 atlases each, target = the T1 of the held-out subject, ground truth =
its `labels.DKT31.manual+aseg` labels; mean Dice over the labels of the subject (106-107):

| subject | CPU | GPU | GPU - CPU | | subject | CPU | GPU | GPU - CPU |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.7233 | 0.7223 | -0.0010 | | 9 | 0.7511 | 0.7502 | -0.0009 |
| 3 | 0.7390 | 0.7395 | +0.0004 | | 11 | 0.7367 | 0.7368 | +0.0001 |
| 5 | 0.7416 | 0.7414 | -0.0001 | | 13 | 0.7269 | 0.7295 | +0.0026 |
| 7 | 0.7270 | 0.7267 | -0.0002 | | 15 | 0.7357 | 0.7410 | +0.0053 |

Mean over the subjects: CPU 0.7351, GPU 0.7359; paired difference GPU - CPU **+0.0008, 95% CI [-0.0010, +0.0026]** (t distribution, n = 8;
bootstrap over subjects [-0.0004, +0.0023]; paired t-test p = 0.34).  With `gpu_numerics=1`: GPU mean 0.7360, difference +0.0009, 95% CI
[-0.0011, +0.0028].  The final parcellations of the GPU and the CPU runs differ in 1.2% of the volume on average (1.0-2.2%), mean
Dice between them 0.926.  Time per subject: CPU 208-242 s, GPU about 8 s (the registration takes longer here, 6.6 s, because the
images are similar and the optimiser iterates longer).

### Registration quality (`bench/reg_scores`, NiftyReg's own measures, warped atlas vs target, 3 targets x 20 atlases)

| | NCC | LNCC (5 mm) | NMI | voxels with Jacobian <= 0 |
|---|---|---|---|---|
| CPU `reg_f3d` | 0.8060 | 0.6328 | 1.1275 | 0.36% |
| GPU F3D | 0.8012 | 0.6327 | 1.1267 | 0.41% |

Paired GPU - CPU per atlas: LNCC -0.0001 (median +0.0019, sd 0.088), NMI -0.0007 (median +0.0003, sd 0.016), NCC -0.0048 (median +0.0008, sd 0.068);
the spread comes from the atlases that ended in a different local optimum.
