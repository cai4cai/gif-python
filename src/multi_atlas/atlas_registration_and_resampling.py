import os
import time
import nibabel as nib
import numpy as np

from src.multi_atlas.atlas_propagation import register_atlas_to_img, propagate_atlas_seg
from src.multi_atlas.utils import compute_disp_from_cpp
from src.multi_atlas.utils import get_lncc_distance

from src.utils.definitions import USE_OLD_RESULTS, WEIGHTS_TEMPERATURE


def warp_atlas_and_calc_similarity_weights(atlas_folder,
                                           save_folder,
                                           img_path,
                                           mask_path
                                           ):

    atlas_name = os.path.split(atlas_folder)[1]

    atlas_img_path = os.path.join(atlas_folder, 'srr.nii.gz')
    atlas_seg_path = os.path.join(atlas_folder, 'parcellation.nii.gz')
    atlas_mask_path = os.path.join(atlas_folder, 'mask.nii.gz')

    save_folder_atlas = os.path.join(save_folder, atlas_name)

    # List of files created by reg_aladin, reg_f3d, reg_resample
    affine_path = os.path.join(save_folder_atlas, 'affine.txt')
    affine_warped_atlas_img_path = os.path.join(save_folder_atlas, 'affine_warped_atlas_img.nii.gz')
    cpp_path = os.path.join(save_folder_atlas, 'cpp.nii.gz')
    warped_atlas_img_path = os.path.join(save_folder_atlas, 'warped_atlas_img.nii.gz')
    warped_atlas_seg_path = os.path.join(save_folder_atlas, 'warped_atlas_seg.nii.gz')
    disp_field_path = os.path.join(save_folder_atlas, 'disp.nii.gz')
    lncc_distance_path = os.path.join(save_folder_atlas, 'lncc_distance.nii.gz')
    weights_path = os.path.join(save_folder_atlas, 'weights.nii.gz')

    # paths to NOT delete after registration
    to_not_remove = [
        warped_atlas_img_path,
        lncc_distance_path,
        disp_field_path,
        weights_path,
        cpp_path

    ]
    to_not_remove_which_starts_with = [
        "warped_atlas_seg",
        #"warped_atlas_img_linear_interp",
    ]

    # Compute the warped atlas (image and segmentation)

    # check if the registration files already exist
    if USE_OLD_RESULTS and all([os.path.exists(f) for f in [cpp_path, warped_atlas_img_path, warped_atlas_seg_path]]):
        print("Found registration files... Skip registration...")

    else:
        time_0_reg = time.time()

        if not os.path.exists(save_folder_atlas):
            os.mkdir(save_folder_atlas)

        # Register the atlas image to the image, then resample the atlas image in the image space
        affine_params_path, cpp_params_path = register_atlas_to_img(
                                                                    img_path=img_path,
                                                                    mask_path=mask_path,
                                                                    atlas_img_path=atlas_img_path,
                                                                    atlas_mask_path=atlas_mask_path,
                                                                    affine_path=affine_path,
                                                                    affine_warped_atlas_img_path=affine_warped_atlas_img_path,
                                                                    cpp_path=cpp_path,
                                                                    warped_atlas_img_path=warped_atlas_img_path,
                                                                )

        # Transform and resample the atlas segmentation in the image space
        warped_atlas_seg_path = propagate_atlas_seg(
                                                    atlas_seg_path=atlas_seg_path,
                                                    img_path=img_path,
                                                    cpp_path=cpp_params_path,
                                                    warped_atlas_seg_path=warped_atlas_seg_path,
                                                )

        print(f"Registration and resampling of {atlas_name} completed after {time.time() - time_0_reg:.3f} seconds")

    # Compute the LNCC distance based weights for the GIF-like fusion
    # check if the weights already exists

    if USE_OLD_RESULTS and os.path.exists(weights_path):
        print("Found weights file... Skip LNCC distance calculation...")
    else:
        time_0_heat = time.time()

        warped_atlas_nii = nib.load(warped_atlas_img_path)

        compute_disp_from_cpp(cpp_path, img_path, disp_field_path)

        # get the linearly interpolated warped atlas
        # the linear interpolation preserves the edges of the masked images better than cubic spline interpolation
        warped_atlas_linear_interp_nii = nib.load(warped_atlas_img_path.replace(".nii.gz", "_linear_interp.nii.gz"))
        warped_atlas_linear_interp = warped_atlas_linear_interp_nii.get_fdata(dtype=np.float32)

        # compute LNCC distance
        lncc_distance = get_lncc_distance(
            image=nib.load(img_path).get_fdata(dtype=np.float32),
            mask=nib.load(mask_path).get_fdata(dtype=np.float16).astype(np.uint8),
            atlas_warped_image=warped_atlas_linear_interp,
            save_folder_path=save_folder_atlas,
            affine=warped_atlas_nii.affine,
            spacing=warped_atlas_nii.header.get_zooms()[0:3],
        )

        # # save the lncc distance
        # lncc_distance_nii = nib.Nifti1Image(lncc_distance, warped_atlas_nii.affine)
        # nib.save(lncc_distance_nii, lncc_distance_path)

        # calculate weights
        weights = np.exp(-lncc_distance / WEIGHTS_TEMPERATURE)

        # save the weights
        weights_nii = nib.Nifti1Image(weights, warped_atlas_nii.affine)
        nib.save(weights_nii, weights_path)

        print(f"Weights calculation for {atlas_name} completed after {time.time() - time_0_heat:.3f} seconds")

    # Remove temporary files
    for f_n in os.listdir(save_folder_atlas):
        p = os.path.join(save_folder_atlas, f_n)
        if p not in to_not_remove and not any([p.split(os.sep)[-1].startswith(s) for s in to_not_remove_which_starts_with]):
            os.system('rm %s' % p)

    return warped_atlas_seg_path, weights_path
