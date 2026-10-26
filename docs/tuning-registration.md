# Tuning and checking the registration

The quality of the segmentation depends mostly on how well the atlases are registered to the target.
All registration parameters can be changed without touching the code, and the defaults are unchanged
(they are the values of the former `src/utils/definitions.py`; several differ from the C++ GIF, see below).

```bash
gif-python segment ... --config reg_f3d_lncc=7 --config reg_f3d_be=0.01
gif-python segment --print-config            # all parameters and their current values
```

From Python, pass a `GifConfig`: `multi_atlas_segmentation(..., config=GifConfig(reg_f3d_lncc=7.0))`.

## The parameters

| `GifConfig` field | NiftyReg option | Default | What it does |
|---|---|---|---|
| `reg_f3d_lncc` | `reg_f3d --lncc` | 5.0 | Standard deviation of the Gaussian kernel that defines the neighbourhood of the local normalised cross-correlation (LNCC) similarity (units as in NiftyReg). A larger kernel compares larger neighbourhoods: more robust to noise and to slowly varying intensity differences between scans, less sensitive to small structures. A small kernel is more local but noisier and uninformative in flat regions. This is the first parameter to try when images differ in contrast from the atlases. |
| `reg_f3d_be` | `-be` | 0.005 | Weight of the bending energy penalty (second derivatives of the transformation). Larger values give smoother transformations that are less likely to fold, but cannot follow small anatomical differences. |
| `reg_f3d_jl` | `-jl` | 0.0001 | Weight of the penalty on the logarithm of the Jacobian determinant. It discourages folding (non-positive Jacobian determinants) and compression. |
| `reg_f3d_maxit` | `-maxit` | 250 | Maximal number of iterations at the final level. More iterations give the optimiser more time; the gain is usually small once the cost no longer decreases. |
| `reg_f3d_ln`, `reg_f3d_lp` | `-ln`, `-lp` | 4, 3 | Number of pyramid levels, and how many of them are run, coarse to fine. With 4 and 3 the finest level is not run. |
| `reg_f3d_grid_spacing` | `-sx` | -5.0 | Final control point spacing; negative values are in voxels (here 5 voxels), positive values in mm. Coarser grids give smoother transformations and faster runs. |
| `reg_aladin_lp` | `reg_aladin -lp` | 2 | Levels of the affine registration. |
| `lncc_sigma` | (Python) | -2.5 (x3) | Kernel of the LNCC that turns the registered atlas into a fusion weight. Negative values are millimetres and are divided by the voxel size; positive values are voxels. |
| `weights_temperature` | (Python) | 0.15 | Temperature of `exp(-(1 - LNCC) / T)`. Smaller values concentrate the fusion on the best-registered atlases. |
| `seg_em_mrf_beta` and other `seg_em_*` | `seg_EM -mrf_beta` ... | 0.1 | Final EM refinement of the tissues. |

Some defaults differ from the C++ GIF (for example the MRF weight 0.1 here and 0.15 in C++ GIF, and the LNCC
kernel of the weights). They were not changed; aligning them is a separate decision.

Change one parameter at a time, use a few representative subjects, and judge the result with the checks
below and, if you have reference labels, with Dice scores (for Mindboggle-101 `scripts/mindboggle42/evaluate_mb42.py`).
The values suggested above (a larger `reg_f3d_lncc`, a larger `reg_f3d_be`) are things to try, not validated recommendations.
The target and the atlases should be prepared in the same way (both brain extracted or both with skull, or
pass masks).

## The NiftyReg warnings

During `reg_f3d` you will usually see lines like

```
[NiftyReg WARNING] The current level reached the maximum number of iteration
[NiftyReg WARNING] Function: reg_f3d<T>::ComputeJacobianBasedPenaltyTerm()
[NiftyReg WARNING] The folding correction scheme failed
```

They are informational messages of NiftyReg and are not failures by themselves:

* *reached the maximum number of iteration*: the optimiser of that pyramid level stopped because of the
  iteration limit (`reg_f3d_maxit`) and not because the cost stopped improving. With the default of 250 this
  is common.
* *The folding correction scheme failed*: while evaluating the Jacobian-based penalty, NiftyReg tried to
  repair a folded part of the control point grid and could not do it completely in that iteration. The
  optimisation continues, and the penalty (`reg_f3d_jl`, `reg_f3d_be`) keeps working against folding.

The C++ GIF uses the same NiftyReg class (`reg_f3d<float>`), so the messages also appear there. In the reference run used
to check the refactoring (one IXI target, three OASIS-TRT-20 atlases, NiftyReg 1.5.77) 32 such lines were
printed while every tool exited with status 0 and the result was reproducible.
`gif-python` therefore only reports how many warning lines a successful call printed; run it with `-vv` to
see them. A call that fails (non-zero exit status) stops the run with an error that contains the command and
the last lines of the output.

The messages become relevant only if the registration result is bad. Then check it as described next.

## Checking the quality of a registration

By default the registration files are deleted after use. Keep them with `--keep-registration-files`
(`keep_registration_files=True`); they are then in `OUT/registered_atlases/<atlas>/`:
`affine.txt`, `affine_warped_atlas_img.nii.gz` (after the affine step), `cpp.nii.gz` (control point grid) and
`warped_atlas_img.nii.gz` (after the non-linear step).

1. **Look at it.** Overlay the warped atlas image on the target, for example
   `itksnap -g target.nii.gz -o OUT/registered_atlases/<atlas>/warped_atlas_img.nii.gz`, and compare
   the contours of ventricles and cortex. Compare with `affine_warped_atlas_img.nii.gz`: the non-linear result
   should fit better.
2. **Measure it.** NiftyReg's `reg_measure` reports similarities between two images:
   ```bash
   reg_measure -ref target.nii.gz -flo OUT/registered_atlases/<atlas>/warped_atlas_img.nii.gz -lncc -nmi -ncc
   reg_measure -ref target.nii.gz -flo OUT/registered_atlases/<atlas>/affine_warped_atlas_img.nii.gz -lncc -nmi -ncc
   ```
   The non-linear values should be better (higher NMI/NCC/LNCC) than the affine ones. Compare atlases with each
   other: an atlas that is much worse than the others contributes little to the fusion anyway.
3. **Check folding.** `reg_jacobian -trans cpp.nii.gz -ref target.nii.gz -jac jac.nii.gz` writes the Jacobian
   determinant; the fraction of voxels with a determinant `<= 0` should be tiny (a small fraction is usually acceptable,
   large folded patches are not):
   ```python
   import nibabel as nib, numpy as np

   jac = nib.load("jac.nii.gz").get_fdata()
   print((jac <= 0).mean())
   ```
4. **Check the output.** `final_parcellation.nii.gz` and `multi_atlas_tissue_seg.nii.gz` over the target image
   show whether the failure comes from the registration or from the tissue step (for example a wrong brain mask).

When the registration files are kept (`--keep-registration-files`) and the same output folder is used again,
they are reused (`use_old_results`), so after changing only a fusion or EM parameter the registrations are not
repeated. After changing a registration parameter, use a new output folder or `--no-reuse`.
