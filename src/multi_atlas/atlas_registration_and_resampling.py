"""Per-atlas processing: registration, resampling of the segmentation and fusion weights."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import nibabel as nib
import numpy as np

from src.multi_atlas.atlas_propagation import (
    get_linearly_interpolated_warped_atlas_img,
    propagate_atlas_seg,
    register_atlas_to_img,
)
from src.multi_atlas.utils import get_lncc_distance
from src.utils.config import GifConfig, default_config
from src.utils.nifti import relax_quaternion_check

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

#: files in the atlas folder that are kept when the temporary files are removed
_KEEP_NAMES = ("lncc_distance.nii.gz", "disp.nii.gz", "weights.nii.gz")
_KEEP_PREFIXES = ("warped_atlas_seg",)


def warp_atlas_and_calc_similarity_weights(
    img_path: PathLike,
    mask_path: Optional[PathLike],
    atlas_name: str,
    atlas_img_path: PathLike,
    atlas_seg_path: PathLike,
    atlas_mask_path: Optional[PathLike],
    save_dir: PathLike,
    config: Optional[GifConfig] = None,
) -> Tuple[str, str]:
    """Register an atlas to the target and compute its similarity weights.

    The atlas image is registered to the target image and resampled into the target space; the same
    transformation is then applied to the atlas segmentation. Finally the similarity weights
    between the registered atlas image and the target image are computed from the LNCC distance.

    Results are written to ``<save_dir>/registered_atlases/<atlas_name>/``; if they are already there
    and ``config.use_old_results`` is set, the corresponding steps are skipped. Besides the warped atlas
    segmentation and the weights, the files of the registration (``affine.txt``, ``cpp.nii.gz``,
    ``warped_atlas_img.nii.gz``, ...) are only kept with ``config.keep_registration_files``.

    :param img_path: path to the target image
    :param mask_path: path to the target mask, can be None
    :param atlas_name: name of the atlas
    :param atlas_img_path: path to the atlas image
    :param atlas_seg_path: path to the atlas segmentation
    :param atlas_mask_path: path to the atlas mask, can be None
    :param save_dir: path to the directory where to save the results
    :param config: pipeline parameters (default: ``default_config()``)
    :return: ``(warped_atlas_seg_path, weights_path)``
    """
    config = config or default_config()
    relax_quaternion_check()

    save_dir_atlas = Path(save_dir) / "registered_atlases" / atlas_name

    # files created by reg_aladin, reg_f3d, reg_resample
    affine_path = save_dir_atlas / "affine.txt"
    affine_warped_atlas_img_path = save_dir_atlas / "affine_warped_atlas_img.nii.gz"
    cpp_path = save_dir_atlas / "cpp.nii.gz"
    warped_atlas_img_path = save_dir_atlas / "warped_atlas_img.nii.gz"
    lin_interp_warped_atlas_img_path = save_dir_atlas / "warped_atlas_img_linear_interp.nii.gz"
    warped_atlas_seg_path = save_dir_atlas / "warped_atlas_seg.nii.gz"
    weights_path = save_dir_atlas / "weights.nii.gz"

    # Compute the warped atlas (image and segmentation) unless the files already exist
    if config.use_old_results and all(f.exists() for f in (cpp_path, warped_atlas_img_path, warped_atlas_seg_path)):
        logger.info("Found registration files for %s... skip registration", atlas_name)
    else:
        time_0_reg = time.time()
        save_dir_atlas.mkdir(parents=True, exist_ok=True)

        # Register the atlas image to the image, then resample the atlas image in the image space
        _, cpp_params_path = register_atlas_to_img(
            img_path=img_path,
            mask_path=mask_path,
            atlas_img_path=atlas_img_path,
            atlas_mask_path=atlas_mask_path,
            affine_path=affine_path,
            affine_warped_atlas_img_path=affine_warped_atlas_img_path,
            cpp_path=cpp_path,
            warped_atlas_img_path=warped_atlas_img_path,
            config=config,
        )

        # Transform and resample the atlas segmentation in the image space
        propagate_atlas_seg(
            atlas_seg_path=atlas_seg_path,
            img_path=img_path,
            cpp_path=cpp_params_path,
            warped_atlas_seg_path=warped_atlas_seg_path,
            config=config,
        )
        logger.info(
            "Registration and resampling of %s completed after %.3f seconds", atlas_name, time.time() - time_0_reg
        )

    # Compute the LNCC distance based weights for the GIF-like fusion
    if config.use_old_results and weights_path.exists():
        logger.info("Found weights file for %s... skip LNCC distance calculation", atlas_name)
    else:
        time_0_heat = time.time()

        warped_atlas_nii = nib.load(str(warped_atlas_img_path))

        # The linear interpolation preserves the edges of the masked images better than cubic spline
        # interpolation: apply the registration to the atlas image again with linear interpolation
        get_linearly_interpolated_warped_atlas_img(
            img_path, atlas_img_path, cpp_path, lin_interp_warped_atlas_img_path, config=config
        )
        warped_atlas_linear_interp = nib.load(str(lin_interp_warped_atlas_img_path)).get_fdata(dtype=np.float32)

        # compute LNCC distance
        lncc_distance = get_lncc_distance(
            image=nib.load(str(img_path)).get_fdata(dtype=np.float32),
            mask=None if not mask_path else nib.load(str(mask_path)).get_fdata(dtype=np.float16).astype(np.uint16),
            atlas_warped_image=warped_atlas_linear_interp,
            spacing=warped_atlas_nii.header.get_zooms()[0:3],
            config=config,
        )

        # calculate and save the weights
        weights = np.exp(-lncc_distance / config.weights_temperature)
        nib.save(nib.Nifti1Image(weights, warped_atlas_nii.affine), str(weights_path))
        logger.info("Weights calculation for %s completed after %.3f seconds", atlas_name, time.time() - time_0_heat)

    # Remove temporary files
    if not config.keep_registration_files:
        for entry in save_dir_atlas.iterdir():
            if entry.is_file() and entry.name not in _KEEP_NAMES and not entry.name.startswith(_KEEP_PREFIXES):
                entry.unlink()

    return str(warped_atlas_seg_path), str(weights_path)
