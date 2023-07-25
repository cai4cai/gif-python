import os

import nibabel as nib
from scipy.ndimage import gaussian_filter
import numpy as np
from src.utils.definitions import (NIFTYREG_PATH, RESAMPLE_METHOD, reg_aladin_LP, reg_f3d_GRID_SPACING, reg_f3d_BE, reg_f3d_LN,
                                   reg_f3d_LP, reg_f3d_JL, reg_f3d_MAXIT, reg_f3d_LNCC, reg_f3d_INTERP, OMP, SIGMA)

def register_and_resample_atlas(
                                image_nii,
                                mask_nii,
                                template_nii,
                                template_seg_nii,
                                template_mask_nii,
                                atlas_seg_onehot_path,
                                warped_altas_seg_onehot_path,
                                warped_atlas_img_path,
                                num_class,
                                save_folder_path,
                                use_affine=True,
                                affine_only=False):
    """
    Summary of what this function does:
    1. register the template to the image
    2. propagate the labels

    The segmentation is used during registration with MSE if seg_nii is not None.
    """

    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    # Register the template to the image
    affine_params_path, cpp_params_path = _register_atlas_to_img(
        image_nii=image_nii,
        mask_nii=mask_nii,
        atlas_nii=template_nii,
        atlas_mask_nii=template_mask_nii,
        warped_atlas_img_path=warped_atlas_img_path,
        save_folder=save_folder_path,
        use_affine=use_affine,
        affine_only=affine_only,
    )

    # Propagate the labels
    warped_atlas_path_or_prob_seg_l_paths, warped_atlas_seg_mask = _propagate_labels(
        num_class=num_class,
        atlas_seg_nii=template_seg_nii,
        image_nii=image_nii,
        cpp_path=cpp_params_path,
        save_folder=save_folder_path,
    )

    return warped_atlas_path_or_prob_seg_l_paths, warped_atlas_seg_mask


def _register_atlas_to_img(image_nii,
                           mask_nii,
                           atlas_nii,
                           atlas_mask_nii,
                           warped_atlas_img_path,
                           save_folder,
                           use_affine,
                           affine_only):
    """
    Affine registration + non-linear registration with stationary velocity fields
    are performed to register the atlas to the image.
    """

    # # TODO: remove this part or make consistent
    # if os.path.isfile(os.path.join(save_folder, 'outputAffine.txt')) \
    #         and os.path.isfile(os.path.join(save_folder, 'cpp.nii.gz')) \
    #         and os.path.isfile(warped_atlas_img_path):
    #     print("Found registration files... Skip registration...")
    #     return os.path.join(save_folder, 'outputAffine.txt'), os.path.join(save_folder, 'cpp.nii.gz')

    # Return the path to the output velocity field returned by NiftyReg
    def save_nifti(volume_np, affine, save_path):
        volume_nii = nib.Nifti1Image(volume_np, affine)
        nib.save(volume_nii, save_path)

    print(f"Use {OMP} subprocesses in each pool")


    img_path = image_nii.file_map['image'].filename
    mask_path = mask_nii.file_map['image'].filename
    atlas_img_path = atlas_nii.file_map['image'].filename
    atlas_mask_path = atlas_mask_nii.file_map['image'].filename

    # Affine registration
    if use_affine or affine_only:

        affine_path = os.path.join(save_folder, 'outputAffine.txt')
        affine_res_path = os.path.join(save_folder, 'affine_warped_atlas_img.nii.gz')

        affine_reg_cmd = (
            f'{NIFTYREG_PATH}/reg_aladin '
            f'-ref "{img_path}" '
            f'-rmask "{mask_path}" '
            f'-flo "{atlas_img_path}" '
            #f'-fmask "{atlas_mask_path}" '
            f'-res "{affine_res_path}" '
            f'-aff "{affine_path}" '
            f'-omp {OMP} '
            f'-lp {reg_aladin_LP} '
            f'-voff '
        )
        print(affine_reg_cmd)
        os.system(affine_reg_cmd)

        if affine_only:
            return affine_path, None

    else:  # no affine transformation
        affine_path = None
        affine_res_path = atlas_img_path
        affine_res_mask_path = atlas_mask_path

    # Registration
    res_path = warped_atlas_img_path
    cpp_path = os.path.join(save_folder, 'cpp.nii.gz')

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
        f'-rmask "{mask_path}" '
        #f'-fmask "{atlas_mask_path}" '
        f'-aff "{affine_path}" '
        f'{reg_options} '
        f'-omp {OMP} '
        f'-res "{res_path}" '  # Filename of the resampled image
        f'-cpp "{cpp_path}" '  # Filename of control point grid [outputCPP.nii]
        f'-voff'
    )

    # print('Non linear registration command line:')
    # print(reg_cmd)
    os.system(reg_cmd)

    # apply the registration to the atlas image again using reg_resample
    reg_resample_cmd = (
        f'{NIFTYREG_PATH}/reg_resample '
        f'-ref "{img_path}" '
        f'-flo "{atlas_img_path}" '
        f'-trans "{cpp_path}" '
        f'-inter 1 '
        f'-res "{res_path.replace(".nii.gz", "_linear_interp.nii.gz")}" '
        f'-voff '

    )
    os.system(reg_resample_cmd)

    return affine_path, cpp_path


def _propagate_labels(num_class,
                      atlas_seg_nii,
                      image_nii,
                      cpp_path,
                      save_folder,
                      ):

    image_path = os.path.join(save_folder, 'img.nii.gz')
    nib.save(image_nii, image_path)

    # Smooth labels and save them separately
    atlas_seg = atlas_seg_nii.get_fdata().astype(np.uint8)


    if RESAMPLE_METHOD == 0:
        # Save the atlas seg as input for reg_resample
        atlas_seg_path = os.path.join(save_folder, "atlas_seg.nii.gz")
        atlas_seg_nii = nib.Nifti1Image(atlas_seg, atlas_seg_nii.affine)
        nib.save(atlas_seg_nii, atlas_seg_path)

        warped_atlas_seg_path = os.path.join(save_folder, f"warped_atlas_seg.nii.gz")

        # Warp the atlas seg using reg_resample and save the warped file
        # Warp the atlas seg given a pre-computed transformation (vel) and save it
        cmd = (
            f'{NIFTYREG_PATH}/reg_resample '
            f'-ref "{image_path}" '
            f'-flo "{atlas_seg_path}" '
            f'-trans "{cpp_path}" '
            f'-res "{warped_atlas_seg_path}" '
            f'-inter 0 '
            f'-omp {OMP} '
            f'-voff '
        )
        os.system(cmd)

        # Load the warped atlas seg volume
        warped_atlas_seg_nii = nib.load(warped_atlas_seg_path)
        warped_atlas_seg = warped_atlas_seg_nii.get_fdata().astype(np.uint8)

        # Create the background segmentation
        warped_atlas_seg_mask = np.zeros_like(warped_atlas_seg)
        warped_atlas_seg_mask[warped_atlas_seg > 0] = 1

        return warped_atlas_seg_path, warped_atlas_seg_mask


    elif RESAMPLE_METHOD == 1:
        warped_atlas_seg_l_paths = [os.path.join(save_folder, f"warped_atlas_seg_{l}.nii.gz") for l in range(num_class)]
        for l in range(num_class):
            print(l)
            atlas_seg_l = np.zeros_like(atlas_seg)
            atlas_seg_l[atlas_seg==l] = 1

            # smooth the atlas
            if SIGMA>0:
                atlas_seg_l = gaussian_filter(atlas_seg_l, sigma=SIGMA, order=0, mode='nearest')

            # save atlas seg as input for reg_resample
            atlas_seg_l_path = os.path.join(save_folder, f"atlas_seg_{l}.nii.gz")
            atlas_seg_l_nii = nib.Nifti1Image(atlas_seg_l, atlas_seg_nii.affine)
            nib.save(atlas_seg_l_nii, atlas_seg_l_path)

            # where should reg_resample save the warped files
            warped_atlas_seg_l_path = warped_atlas_seg_l_paths[l]


            # Warp the atlas seg given a pre-computed transformation (vel) and save it
            cmd = (
                f'{NIFTYREG_PATH}/reg_resample '
                f'-ref "{image_path}" '
                f'-flo "{atlas_seg_l_path}" '
                f'-trans "{cpp_path}" '
                f'-res "{warped_atlas_seg_l_path}" '
                f'-inter 1 '
                f'-omp {OMP}'
                f'-voff '
            )
            os.system(cmd)

        sum_warped_atlas_segs = None
        for l in range(num_class):
            # Load and return the warped atlas proba numpy array
            warped_altas_seg_l_nii = nib.load(warped_atlas_seg_l_paths[l])
            warped_altas_seg_l = warped_altas_seg_l_nii.get_fdata().astype(np.float32)  # H x W x D; is smoothed

            if l == 0:
                sum_warped_atlas_segs = np.zeros_like(warped_altas_seg_l)
            else:
                sum_warped_atlas_segs += warped_altas_seg_l


        # reg_resample pads all images with 0. Include these regions as background by setting them to 1 in the background segmentation
        # sum_warped_atlas_segs is 0 for pixels that were added by padding
        warped_altas_seg_0_nii = nib.load(warped_atlas_seg_l_paths[0])
        warped_altas_seg_0 = warped_altas_seg_0_nii.get_fdata().astype(np.float32)
        warped_altas_seg_0[sum_warped_atlas_segs == 0] = 1.  # H x W x D x C ; label 0 map is now 1 where no label was present (which is where reg_resample padded with 0)
        warped_altas_seg_0_nii = nib.Nifti1Image(warped_altas_seg_0, warped_altas_seg_0_nii.affine)
        nib.save(warped_altas_seg_0_nii, warped_atlas_seg_l_paths[0])

        sum_warped_atlas_segs[sum_warped_atlas_segs == 0] = 1  # Update the sum as well for normalization

        # get probabilistic atlas by normalizing across label dimension
        # additionally calculate the atlas mask (where background has the largest probability)
        for l in range(num_class):
            warped_altas_seg_l_nii = nib.load(warped_atlas_seg_l_paths[l])
            warped_altas_seg_l = warped_altas_seg_l_nii.get_fdata().astype(np.float32)  # H x W x D; is smoothed
            warped_altas_seg_normalized_l = warped_altas_seg_l / sum_warped_atlas_segs
            warped_altas_seg_normalized_l_nii = nib.Nifti1Image(warped_altas_seg_normalized_l, atlas_seg_nii.affine)
            nib.save(warped_altas_seg_normalized_l_nii, warped_atlas_seg_l_paths[l])

            if l == 0:
                warped_altas_seg_0 = warped_altas_seg_l
                warped_atlas_seg_mask = np.zeros_like(warped_altas_seg_0)
            else:
                warped_atlas_seg_mask[warped_altas_seg_l > warped_altas_seg_0] = 1

        return warped_atlas_seg_l_paths, warped_atlas_seg_mask