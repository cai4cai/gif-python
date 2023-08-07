import os
from src.utils.definitions import (NIFTYREG_PATH, RESAMPLE_METHOD, reg_aladin_LP, reg_f3d_GRID_SPACING, reg_f3d_BE, reg_f3d_LN,
                                   reg_f3d_LP, reg_f3d_JL, reg_f3d_MAXIT, reg_f3d_LNCC, reg_f3d_INTERP, OMP, SIGMA)


def register_atlas_to_img(img_path,
                          mask_path,
                          atlas_img_path,
                          atlas_mask_path,
                          affine_path,
                          affine_warped_atlas_img_path,
                          cpp_path,
                          warped_atlas_img_path,
                          ):
    """
    Affine registration + non-linear registration with stationary velocity fields
    are performed to register the atlas to the image.
    """

    print(f"Use {OMP} subprocesses in each pool")

    # Affine registration
    affine_reg_cmd = (
        f'{NIFTYREG_PATH}/reg_aladin '
        f'-ref "{img_path}" '
        f'-flo "{atlas_img_path}" '
        f'-res "{affine_warped_atlas_img_path}" '
        f'-aff "{affine_path}" '
        f'-omp {OMP} '
        f'-lp {reg_aladin_LP} '
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
        f'-jl {reg_f3d_JL} '  # Weight of log of the Jacobian determinant penalty term
        f'-be {reg_f3d_BE} '  # Weight of the bending energy (second derivative of the transformation) penalty term
        f'-maxit {reg_f3d_MAXIT} '  # Maximum number of iterations
        f'-ln {reg_f3d_LN} '  # Number of levels
        f'-lp {reg_f3d_LP} '  # Only perform the first levels [ln]
        f'-sx {reg_f3d_GRID_SPACING} '  # Final grid spacing in the x direction, adopted in y and z directions if not specified
        f'--lncc {reg_f3d_LNCC} '  # Standard deviation of the Gaussian kernel.
        f'--interp {reg_f3d_INTERP} '  # Interpolation order (0=NN, 1=linear, 3=cubic)
    )
    reg_cmd = (
        f'{NIFTYREG_PATH}/reg_f3d '
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

    # apply the registration to the atlas image again using reg_resample with linear interpolation
    reg_resample_cmd = (
        f'{NIFTYREG_PATH}/reg_resample '
        f'-ref "{img_path}" '
        f'-flo "{atlas_img_path}" '
        f'-trans "{cpp_path}" '
        f'-inter 1 '
        f'-res "{warped_atlas_img_path.replace(".nii.gz", "_linear_interp.nii.gz")}" '
        f'-voff '

    )

    if os.system(reg_resample_cmd):
        print(f'Error in reg_resample! The command was:\n{reg_resample_cmd}')
        exit(1)

    return affine_path, cpp_path


def propagate_atlas_seg(
                      atlas_seg_path,
                      img_path,
                      cpp_path,
                      warped_atlas_seg_path,
                      ):

    if RESAMPLE_METHOD == 0:

        # Warp the atlas seg using reg_resample and save the warped file
        cmd = (
            f'{NIFTYREG_PATH}/reg_resample '
            f'-ref "{img_path}" '
            f'-flo "{atlas_seg_path}" '
            f'-trans "{cpp_path}" '
            f'-res "{warped_atlas_seg_path}" '
            f'-inter 0 '
            f'-omp {OMP} '
            f'-voff '
        )
        os.system(cmd)

        return warped_atlas_seg_path
