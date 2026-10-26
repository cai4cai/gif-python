"""Registration of an atlas to the target image and propagation of the atlas segmentation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

from src.utils.config import GifConfig, default_config
from src.utils.process import run_tool

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def register_atlas_to_img(
    img_path: PathLike,
    mask_path: Optional[PathLike],
    atlas_img_path: PathLike,
    atlas_mask_path: Optional[PathLike],
    affine_path: PathLike,
    affine_warped_atlas_img_path: PathLike,
    cpp_path: PathLike,
    warped_atlas_img_path: PathLike,
    config: Optional[GifConfig] = None,
) -> Tuple[PathLike, PathLike]:
    """Register an atlas image to the target image (affine ``reg_aladin``, then ``reg_f3d``).

    :param img_path: target image (reference)
    :param mask_path: mask of the target image, or ``None``
    :param atlas_img_path: atlas image (floating)
    :param atlas_mask_path: mask of the atlas image, or ``None``
    :param affine_path: where to write the affine transformation
    :param affine_warped_atlas_img_path: where to write the affinely resampled atlas image
    :param cpp_path: where to write the control point grid of the non-linear transformation
    :param warped_atlas_img_path: where to write the non-linearly resampled atlas image
    :param config: pipeline parameters (default: ``default_config()``)
    :return: ``(affine_path, cpp_path)``
    :raises ExternalToolError: if a NiftyReg tool fails
    """
    config = config or default_config()
    logger.info("Use %s OpenMP threads in each NiftyReg process", config.omp)

    # Affine registration
    affine_reg_cmd = [
        config.binary("reg_aladin"),
        "-ref", img_path,
        "-flo", atlas_img_path,
        "-res", affine_warped_atlas_img_path,
        "-aff", affine_path,
        "-omp", config.omp,
        "-lp", config.reg_aladin_lp,
        "-voff",
    ]  # fmt: skip
    # masks are optional
    if mask_path:
        affine_reg_cmd += ["-rmask", mask_path]
    if atlas_mask_path:
        affine_reg_cmd += ["-fmask", atlas_mask_path]
    logger.info("Affine registration of %s", Path(atlas_img_path).name)
    run_tool(affine_reg_cmd)

    # Non-linear registration
    reg_cmd = [
        config.binary("reg_f3d"),
        "-ref", img_path,
        "-flo", atlas_img_path,
        "-aff", affine_path,
        "-jl", config.reg_f3d_jl,  # weight of the log of the Jacobian determinant penalty term
        "-be", config.reg_f3d_be,  # weight of the bending energy penalty term
        "-maxit", config.reg_f3d_maxit,  # maximum number of iterations
        "-ln", config.reg_f3d_ln,  # number of levels
        "-lp", config.reg_f3d_lp,  # only perform the first levels [ln]
        "-sx", config.reg_f3d_grid_spacing,  # final grid spacing (adopted for y and z)
        "--lncc", config.reg_f3d_lncc,  # standard deviation of the LNCC Gaussian kernel
        "--interp", config.reg_f3d_interp,  # interpolation order (0=NN, 1=linear, 3=cubic)
        "-omp", config.omp,
        "-res", warped_atlas_img_path,
        "-cpp", cpp_path,  # control point grid
        "-voff",
    ]  # fmt: skip
    if mask_path:
        reg_cmd += ["-rmask", mask_path]
    if atlas_mask_path:
        reg_cmd += ["-fmask", atlas_mask_path]
    logger.info("Non-linear registration of %s", Path(atlas_img_path).name)
    run_tool(reg_cmd)

    return affine_path, cpp_path


def propagate_atlas_seg(
    atlas_seg_path: PathLike,
    img_path: PathLike,
    cpp_path: PathLike,
    warped_atlas_seg_path: PathLike,
    config: Optional[GifConfig] = None,
) -> PathLike:
    """Resample an atlas segmentation into the target space with the non-linear transformation.

    :return: ``warped_atlas_seg_path``
    :raises NotImplementedError: for ``config.resample_method != 0`` (only nearest-neighbour
        resampling of the label map is implemented)
    :raises ExternalToolError: if ``reg_resample`` fails
    """
    config = config or default_config()
    if config.resample_method != 0:
        raise NotImplementedError(
            f"resample_method={config.resample_method} is not implemented; only 0 (nearest neighbour) is supported"
        )
    run_tool(
        [
            config.binary("reg_resample"),
            "-ref",
            img_path,
            "-flo",
            atlas_seg_path,
            "-trans",
            cpp_path,
            "-res",
            warped_atlas_seg_path,
            "-inter",
            0,
            "-omp",
            config.omp,
            "-voff",
        ]  # fmt: skip
    )
    return warped_atlas_seg_path


def get_linearly_interpolated_warped_atlas_img(
    img_path: PathLike,
    atlas_img_path: PathLike,
    cpp_path: PathLike,
    lin_interp_warped_atlas_img_path: PathLike,
    config: Optional[GifConfig] = None,
) -> PathLike:
    """Apply the registration to the atlas image again, with linear interpolation.

    The linear interpolation preserves the edges of masked images better than cubic spline
    interpolation, which avoids edge artifacts in the LNCC distance calculation.

    :param img_path: path to the target image
    :param atlas_img_path: path to the atlas image
    :param cpp_path: path to the control point grid
    :param lin_interp_warped_atlas_img_path: where to write the result
    :param config: pipeline parameters (default: ``default_config()``)
    :return: ``lin_interp_warped_atlas_img_path``
    :raises ExternalToolError: if ``reg_resample`` fails
    """
    config = config or default_config()
    run_tool(
        [
            config.binary("reg_resample"),
            "-ref",
            img_path,
            "-flo",
            atlas_img_path,
            "-trans",
            cpp_path,
            "-inter",
            1,
            "-res",
            lin_interp_warped_atlas_img_path,
            "-voff",
        ]  # fmt: skip
    )
    return lin_interp_warped_atlas_img_path
