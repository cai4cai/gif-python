# Provenance

This branch (`gifpy-gpu`) was produced by a clean-room "seeding" step.  It contains only code that is either

* **A** - a port / reimplementation of functionality of NiftyReg or NiftySeg (both BSD-3-Clause, see `NOTICE`), or
* **B** - original GPU infrastructure (batching, memory pools, streams, gzip writer, drivers, build files).

Everything that is derived from, or whose behaviour is specified by, the C++ source of the GIF segmentation tool (the
Apache-2.0 + Commons Clause licensed `seg_GIF` / `_seg_GIF.cpp` code base) was **excluded**, and the history of the original working branch
is **not** part of this branch (the branch is an orphan with a fresh root).  Excluded areas: group affine averaging, the
propagation / label fusion stage, the tissue prior and brain-mask construction, the parcellation and volume computation, the labels
XML handling and the end-to-end pipeline.  Comments were scrubbed of
references to that code; remaining comments cite NiftyReg / NiftySeg functions only.

License: BSD-3-Clause (`LICENSE`).  The copyright line is a placeholder, `Copyright (c) <year> <holders to be decided>`: **the copyright
holders have not been decided by the seeding step** and must be filled in by the owners before publication.

## Files

| File(s) | Cat. | Upstream function(s) ported | Why clean |
|---|---|---|---|
| `gpu/aladin/aladin.cu` | A+B | NiftyReg `reg_downsampleImage`, `_reg_set_active_blocks`, `reg_affine_getDeformationField`, `reg_resampleImage`, `block_matching_method`, `optimize*` (LTS), `reg_aladin::Run` iteration plan, `reg_mat44_*` | Port of reg_aladin_sym stages (comments cite only NiftyReg functions); batching, stream/memory-pool handling and kernel layout are original. Contains no group-average or database logic. |
| `gpu/aladin/aladin.h` | B | API of the port; defaults are reg_aladin's defaults (`_reg_aladin.cpp`) | Parameter struct with NiftyReg defaults; no application-specific settings. |
| `gpu/aladin/mat4.h` | A | `nifti_mat44_inverse`, `reg_mat44_logm/expm/avg2` (NiftyReg `_reg_maths*`) | Small-matrix maths replacing Eigen; tested against the NiftyReg functions (`maths_check`). The multi-matrix trimmed average was dropped. |
| `gpu/aladin/Makefile, .gitignore` | B | - | Build files. |
| `gpu/aladin/README.md` | B | - | Written for this branch. |
| `gpu/aladin/test/aladin_cli.cu` | B | - | Benchmark driver for the port (raw image inputs, matrix text outputs); group average removed. |
| `gpu/aladin/test/maths_check.cpp` | A | NiftyReg `reg_mat44_logm/expm/avg2`, `nifti_mat44_inverse` | Compares mat4.h with the NiftyReg library on random matrices. |
| `gpu/aladin/test/stage_check.cpp` | A | NiftyReg `AladinContent`, `block_matching_method`, `optimize`, `reg_resampleImage` | Feeds GPU intermediates to NiftyReg functions. |
| `gpu/aladin/test/cpu_trace.cpp` | A | NiftyReg `reg_aladin_sym<float>` | NiftyReg reference run with per-iteration trace; only NiftyReg API calls. |
| `gpu/aladin/tools/prep_data.py` | B | - | NIfTI to raw converter for the driver. |
| `gpu/f3d/include/f3d.h` | B | `reg_f3d` setters | Public API; defaults are reg_f3d's defaults (`_reg_f3d.cpp`, `_reg_base.cpp`). |
| `gpu/f3d/include/f3d_internal.h, f3d_state.h, f3d_util.h` | B | `reg_optimiser`/`reg_f3d` bookkeeping structures | Internal declarations of the port. |
| `gpu/f3d/include/f3d_geom.h` | A | NiftyReg geometry helpers (`reg_getDeformationFromDisplacement`, mat44 handling, grid geometry of `reg_f3d`) | Host geometry helpers mirroring NiftyReg v1.5.77. |
| `gpu/f3d/include/f3d_basis.h, f3d_basis_host.h, f3d_refine_gen.cuh` | A | NiftyReg `_reg_splineBasis.cpp` basis tables, `reg_spline_refineControlPointGrid3D` | Constant tables / expressions extracted from NiftyReg. |
| `gpu/f3d/src/f3d_conv.cu` | A | `reg_createImagePyramid`, `reg_tools_kernelConvolution`, `reg_tools_rescaleImage` | Pyramid, convolution and rescale. |
| `gpu/f3d/src/f3d_transform.cu` | A | `reg_spline_getDeformationField3D`, `reg_resampleImage`, `reg_getImageGradient`, `reg_spline_refineControlPointGrid`, `reg_affine_getDeformationField` | B-spline transform and resampling. |
| `gpu/f3d/src/f3d_lncc.cu, f3d_lncc_stream.cu, f3d_lncc_fast.cu, f3d_lncc_common.cuh` | A+B | `reg_lncc::GetSimilarityMeasureValue`, `GetVoxelBasedSimilarityMeasureGradient` | LNCC port (exact emulation and fast kernels); fused streaming layout is original. |
| `gpu/f3d/src/f3d_gradient.cu` | A | `reg_voxelCentric2NodeCentric`, gradient smoothing (`reg_tools_kernelConvolution`) | Gradient conversion. |
| `gpu/f3d/src/f3d_regul.cu` | A | `reg_spline_approxBendingEnergy*`, `reg_spline_approxLinearEnergy*`, `reg_spline_getJacobianPenalty*`, `reg_spline_correctFolding` | Penalty terms and folding correction. |
| `gpu/f3d/src/f3d_optim.cu` | A | `reg_optimiser`, `reg_conjugateGradient`, `reg_f3d::NormaliseGradient` | Device-side optimiser control. |
| `gpu/f3d/src/f3d_batch.cu` | A+B | level loop and weight scaling of `reg_f3d::Run`/`_reg_f3d.cpp` | Batching over templates, stream groups, level orchestration. |
| `gpu/f3d/Makefile, .gitignore` | B | - | Build files. |
| `gpu/f3d/README.md` | B | - | Written for this branch. |
| `gpu/f3d/test/test_stage1.cpp, test_stage3.cpp, test_stage5.cpp, test_warpdf.cpp, test_m3.cpp` | A | NiftyReg library functions listed in the file headers | Compare the GPU stages with NiftyReg on the same inputs. |
| `gpu/f3d/test/emul_be.cpp, emul_be_value.cpp, emul_le.cpp, replica_be.cpp, test_fma_be.cpp, test_fma_le.cpp` | A | `reg_spline_approxBendingEnergy*`, `reg_spline_approxLinearEnergy*` | Host emulations / brute force search of the float operation structure of NiftyReg's code, checked against the library. |
| `gpu/f3d/test/test_stage6.cpp, test_batch.cpp, bench_lncc.cpp, eval_cpp.cpp` | A+B | NiftyReg `reg_f3d`, `reg_lncc` outputs as reference | Full-run comparison with a NiftyReg grid, batch driver, LNCC micro benchmark, LNCC evaluation of a grid with NiftyReg. |
| `gpu/f3d/test/ref_f3d.cpp, ref_f3d_dbg.cpp` | A | NiftyReg `reg_f3d<float>` | NiftyReg reference run (settings in test_config.h); the debug variant traces objective evaluations. |
| `gpu/f3d/test/test_common.h, test_config.h, run_all_tests.sh` | B | - | Test helpers; test_config.h is an arbitrary non-default test configuration (see note 2). |
| `gpu/f3d/tools/f3d_run.cpp` | B | NiftyReg control point image format | Command line driver writing NiftyReg-format grids. |
| `gpu/seg/include/segem.h` | B | `seg_EM` setters | Public API; defaults are the NiftySeg seg_EM application defaults. |
| `gpu/seg/src/internal.cuh, context.cu` | B | - | Buffers, orchestration, timing. Contains no label-map, prior, parcellation or volume code. |
| `gpu/seg/src/em.cu` | A | NiftySeg `seg_EM::Run_EM`, `Maximization`, `Expectation`, `RunMRF3D`, `RunBiasField`, `InitializeAndNormalizeImage/NaNPriors`, `GetBiasCorrected` | Port of the NiftySeg EM, including its zero-prior fallback behaviour and MRF border/aliasing quirks (see note 3). |
| `gpu/seg/src/morph.cu` | A | NiftySeg `Dillate`, `Erosion`, `Close_Forground_ConnectComp`, `GaussianFilter4D_cArray`, mask smoothing of `seg_EM::GetBiasCorrected` | Morphology primitives; the call sequence/radii of any application were removed (radii are parameters). |
| `gpu/seg/tests/test_morph.cu` | A | NiftySeg `Dillate`, `Close_Forground_ConnectComp`, `Erosion` | Random-mask comparison with the NiftySeg library. |
| `gpu/seg/tests/test_em.cu` | A | NiftySeg `seg_EM` class | New synthetic-data comparison with seg_EM. |
| `gpu/seg/Makefile, .gitignore, README.md` | B | - | Build files and documentation. |
| `gpu/common/gzwrite.h, gzwrite.cpp, test_gzwrite.cpp, Makefile, README.md, .gitignore` | B | zlib `deflate` | Parallel multi-member gzip writer and round-trip test; generic I/O infrastructure. |
| `bench/reg_scores.cpp, Makefile, README.md, .gitignore` | A+B | NiftyReg `reg_lncc`, `reg_nmi`, `reg_spline_getDeformationField`, `reg_resampleImage`, Jacobian functions | Registration quality scores computed with NiftyReg's own measure classes. |
| `LICENSE, NOTICE, PROVENANCE.md, README.md` | B | - | Legal and documentation files. |

## Notes on items where the classification needed a judgement

1. **Defaults.** The parameter structs (`aladin::Params`, `f3d::F3DParams`, `segem::EmParams`) were changed to the defaults of
   the corresponding NiftyReg / NiftySeg tools; application-specific settings that the original branch used as defaults were removed.
2. **`gpu/f3d/test/test_config.h`** holds an arbitrary non-default configuration (penalty weights, level counts, iteration cap) used
   identically by the GPU tests and the NiftyReg reference programs so that both sides are compared with the same settings.  The values
   were taken from the original working branch, where they were the settings of an application; they are plain `reg_f3d` option values
   and can be replaced freely.
3. **`gpu/seg/src/em.cu`** handles classes whose prior is zero everywhere ("dormant classes") and iteration shortcuts that follow from this.
   This is NiftySeg behaviour (the `seg_EM` NaN fallback and uniform posterior), verified with `tests/test_em.cu` against the real
   `seg_EM` class (including a class without prior), but the behaviour was originally discovered while handling an application's
   priors.  Reviewers who want a stricter separation may rewrite this part from `_seg_EM.cpp`.
4. **`gpu/seg/src/morph.cu`**: the `Dillate/Erosion/Gaussian` sequence in `launch_smooth_mask` is the mask construction of
   `seg_EM::GetBiasCorrected` in NiftySeg itself (not an application sequence).
5. The code in this branch was written by an earlier working process that also read the GIF sources.  Only code whose behaviour is
   specified by NiftyReg / NiftySeg was kept and its comments were scrubbed, but a legal review of the remaining kernels may still
   want to compare them against the NiftyReg / NiftySeg sources (the A files are ports of those sources by design).

---

# Files added in the gif-python repository (branch `gpu-backend`)

The files above are the seed (categories A/B as explained at the top).  The following files were written for gif-python by a second,
clean-room implementer, who read only gif-python's own Python code and tests, the BSD-3 NiftyReg / NiftySeg sources and documentation,
the seed above, general documentation (NIfTI-1 header specification, CUDA, numpy / scipy documentation) and image data.  The GIF
(KCL-BMEIS/gif) sources were not read.  The behaviour of the new files is that of gif-python's Python pipeline.

| File(s) | Cat. | Written from | Notes |
|---|---|---|---|
| `CMakeLists.txt` | B | - | Build of `gif_gpu` from the seed libraries (compile flags as in the seed Makefiles). |
| `gif/gif_io.h`, `gif/gif_io.cpp` | B | NIfTI-1 header specification (public domain); nibabel's rule "sform if sform_code > 0, else qform" | Minimal reader / header builder.  The quaternion <-> matrix conversions follow the formulas of the NIfTI-1 specification. |
| `gif/gif_stages.h`, `gif/gif_stages.cu`: `Resampler` | A | NiftyReg `reg_resampleImage` / `ResampleImage3D`, `interpLinearKernel`, `interpNearestNeighKernel`, `reg_round`, `reg_mat44_mul` (BSD-3, `reg-lib/cpu/_reg_resampling.cpp`) and the `reg_resample` application (default padding 0, data type of the floating image); gif-python `atlas_registration_and_resampling.py` | Uses the seed's `warp()` for the B-spline deformation field. |
| `gif/gif_stages.cu`: `LnccWeights` | B | gif-python `src/multi_atlas/utils.py` (`get_lncc_distance`, `gaussian_smooth_with_mask`) and `atlas_registration_and_resampling.py` (weights); documented behaviour of `scipy.ndimage.gaussian_filter` (kernel `exp(-0.5 x^2 / sigma^2)` normalised by its sum, mode `reflect`, one float32 output per axis) | See "Doubts" below for the summation order. |
| `gif/gif_stages.cu`: `Fusion` | B | gif-python `src/multi_atlas/multi_atlas_fusion.py` and `multi_atlas_segmentation.py` (probabilities, argmax, tissue prior, tissue segmentation, final parcellation) | The label probabilities are kept sparse (at most one entry per atlas and voxel). |
| `gif/gif_stages.cu`: `labels_from_float`, `label_presence` | B | gif-python (`get_fdata(dtype=np.float16).astype(np.uint16)`, label checks) | |
| `gif/gif_gpu.cpp` | B | gif-python `multi_atlas_segmentation.py`, `GifConfig` defaults; NiftySeg `seg-apps/seg_EM.cpp` (defaults of the application, mask = all voxels when no mask is given, output = posterior of every class) | Orchestration, command line, timing, output files. |
| `README.md` | B | - | |
| `../src/multi_atlas/gpu_backend.py`, `../tests/test_gpu_backend.py`, `../tests/test_gpu_stages.py`, `../scripts/gpu_validation/*` | B | gif-python | Python glue, tests and validation scripts. |

## Doubts about provenance (for the reviewers)

1. The seed itself was produced by a process that had also read GIF sources (note 5 above); this implementer relied on the seed as
   delivered and did not compare it with anything outside the allowed sources.
2. The summation order of scipy's `correlate1d` (centre tap first, then the pairs of taps from the outside inwards, in double
   precision) and numpy's pairwise summation (used for the kernel normalisation and for `np.sum` over the tissue axis) were
   reproduced from memory of their documented / observed behaviour, not from their sources.  They were checked numerically against
   scipy / numpy (`tests/test_gpu_stages.py`: weights agree to 6e-5 relative, 54% of the voxels bit-identical).
3. Class order: gif-python numbers the labels in the iteration order of a Python `set`; the GPU backend uses ascending label order.
   This only matters for exact ties in `np.argmax`.  For the final parcellation, labels of zero probability are tried in class order,
   whereas numpy's `argpartition` returns some other index among tied values.  Both are documented in `README.md`.
4. No GIF code or GIF-derived material was seen by this implementer.
