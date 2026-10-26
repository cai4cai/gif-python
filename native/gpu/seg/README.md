# gpu/seg - NiftySeg EM segmentation and mask morphology (CUDA)

A GPU port of the NiftySeg (BSD-3-Clause, see `NOTICE`) `seg_EM` expectation-maximisation (single modality, Gaussian mixture, MRF,
polynomial bias field of order 1-4) and of the binary-mask routines `Dillate`, `Erosion`, `Close_Forground_ConnectComp` and the
mask smoothing of `seg_EM::GetBiasCorrected` (`Dillate(7)`, `Erosion(3)`, `GaussianFilter4D_cArray(3)`).

## What is ported
| NiftySeg | here |
|---|---|
| `seg_EM::Run_EM` (M step, E step, MRF `RunMRF3D`, `RunBiasField`, stopping rule, `InitializeAndNormalize*`) | `src/em.cu` |
| `seg_EM::GetBiasCorrected` | `launch_bias_corrected`, `launch_smooth_mask` |
| `Dillate`, `Erosion`, `Close_Forground_ConnectComp` | `src/morph.cu` (the hole filling reproduces the exact output of the NiftySeg routine, which differs from a border flood fill on some masks) |

## API (`include/segem.h`)
`EmParams` (defaults = NiftySeg `seg_EM` application defaults), `EmContext(nx, ny, nz, numClasses, params, liveMask)`,
`EmContext::liveClasses(prior, C, N)` (classes with a non-zero prior; classes without prior are handled as `seg_EM` does),
`run(prior [C][N] float, mask [N] uint8, target [N] float)`, `seg()`, `biasCorrected()`, `morphology(mask, dilate_radius, erode_radius)`.

## Build and test
```
export NIFTYSEG_PREFIX=<install prefix of a NiftySeg build with static libraries>
make            # build/libsegem.a
make tests      # build/test_morph, build/test_em
build/test_morph [ncases] [seed]                       # random masks, GPU vs NiftySeg routines (exact comparison)
build/test_em nx ny nz classes dormant beta order seed  # synthetic volume, GPU vs seg_EM
```
`test_em` builds nested ellipsoid classes with a smooth multiplicative bias field and a noisy prior (optionally with one class that has
an all-zero prior), runs NiftySeg's `seg_EM` on the host and the GPU EM, and compares posteriors, hard labels and the bias corrected image.
