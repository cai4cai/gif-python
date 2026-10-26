import os
from src.utils.config import default_config


def register_atlas_to_img(img_path,
                          mask_path,
                          atlas_img_path,
                          atlas_mask_path,
                          affine_path,
                          affine_warped_atlas_img_path,
                          cpp_path,
                          warped_atlas_img_path,
                          config=None,
                          ):
    """
    Affine registration + non-linear registration with stationary velocity fields
    are performed to register the atlas to the image.
    """
    config = config or default_config()
    OMP = config.omp

    print(f"Use {OMP} subprocesses in each pool")

    # Affine registration
    affine_reg_cmd = (
        f'{config.binary("reg_aladin")} '
        f'-ref "{img_path}" '
        f'-flo "{atlas_img_path}" '
        f'-res "{affine_warped_atlas_img_path}" '
        f'-aff "{affine_path}" '
        f'-omp {OMP} '
        f'-lp {config.reg_aladin_lp} '
        f'-voff '
    )

    # masks are optional
    if mask_path:
        affine_reg_cmd += f'-rmask "{mask_path}" '
    if atlas_mask_path:
        affine_reg_cmd += f'-fmask "{atlas_mask_path}" '

    print('Affine registration command:')
    os.system(affine_reg_cmd)


    # Non-linear registration

    reg_options = (
        f'-jl {config.reg_f3d_jl} '  # Weight of log of the Jacobian determinant penalty term
        f'-be {config.reg_f3d_be} '  # Weight of the bending energy (second derivative of the transformation) penalty term
        f'-maxit {config.reg_f3d_maxit} '  # Maximum number of iterations
        f'-ln {config.reg_f3d_ln} '  # Number of levels
        f'-lp {config.reg_f3d_lp} '  # Only perform the first levels [ln]
        f'-sx {config.reg_f3d_grid_spacing} '  # Final grid spacing in the x direction, adopted in y and z directions if not specified
        f'--lncc {config.reg_f3d_lncc} '  # Standard deviation of the Gaussian kernel.
        f'--interp {config.reg_f3d_interp} '  # Interpolation order (0=NN, 1=linear, 3=cubic)
    )
    reg_cmd = (
        f'{config.binary("reg_f3d")} '
        f'-ref "{img_path}" '
        f'-flo "{atlas_img_path}" '
        f'-aff "{affine_path}" '
        f'{reg_options} '
        f'-omp {OMP} '
        f'-res "{warped_atlas_img_path}" '  # Filename of the resampled image
        f'-cpp "{cpp_path}" '  # Filename of control point grid [outputCPP.nii]
        f'-voff '
    )

    # masks are optional
    if mask_path:
        reg_cmd += f'-rmask "{mask_path}" '
    if atlas_mask_path:
        reg_cmd += f'-fmask "{atlas_mask_path}" '

    print('Non linear registration command:')
    print(reg_cmd)
    os.system(reg_cmd)

    return affine_path, cpp_path


def propagate_atlas_seg(
                      atlas_seg_path,
                      img_path,
                      cpp_path,
                      warped_atlas_seg_path,
                      config=None,
                      ):
    config = config or default_config()
    OMP = config.omp

    if config.resample_method == 0:

        # Warp the atlas seg using reg_resample and save the warped file
        reg_resample_cmd = (
            f'{config.binary("reg_resample")} '
            f'-ref "{img_path}" '
            f'-flo "{atlas_seg_path}" '
            f'-trans "{cpp_path}" '
            f'-res "{warped_atlas_seg_path}" '
            f'-inter 0 '
            f'-omp {OMP} '
            f'-voff '
        )

        if os.system(reg_resample_cmd):
            print(f'Error in reg_resample! The command was:\n{reg_resample_cmd}')
            exit(1)

        return warped_atlas_seg_path


def get_linearly_interpolated_warped_atlas_img(img_path, atlas_img_path, cpp_path, lin_interp_warped_atlas_img_path,
                                               config=None):
    """
    Apply the registration to the atlas image again using reg_resample with linear interpolation
    The linear interpolation preserves the edges of the masked images better than cubic spline interpolation
    This is done to avoid edge artifacts in the LNCC distance calculation

    :param img_path: path to the target image
    :param atlas_img_path: path to the atlas image
    :param cpp_path: path to the cpp file
    :param lin_interp_warped_atlas_img_path: path to the linearly interpolated warped atlas image
    :param config: GifConfig (default: ``default_config()``)

    """
    config = config or default_config()
    reg_resample_cmd = (
        f'{config.binary("reg_resample")} '
        f'-ref "{img_path}" '
        f'-flo "{atlas_img_path}" '
        f'-trans "{cpp_path}" '
        f'-inter 1 '
        f'-res "{lin_interp_warped_atlas_img_path}" '
        f'-voff '

    )

    if os.system(reg_resample_cmd):
        print(f'Error in reg_resample! The command was:\n{reg_resample_cmd}')
        exit(1)

    return lin_interp_warped_atlas_img_path
