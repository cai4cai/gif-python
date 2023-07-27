import os
import time
import nibabel as nib
import numpy as np

from src.multi_atlas.atlas_propagation import register_atlas_to_img, propagate_atlas_seg
from src.multi_atlas.utils import get_lncc_distance

from src.utils.definitions import USE_OLD_RESULTS, WEIGHTS_TEMPERATURE


def warp_atlas_and_calc_similarity_weights(atlas_dir,
                                           img_path,
                                           mask_path,
                                           save_dir,
                                           ):
    """
    Register the atlas image to the target image, then resample in the target image space, then use same transform
    to transform and resample the atlas segmentation in the target space.
    Finally, compute the similarity weights between each registered atlas image and the target image.

    :param atlas_dir: path to the atlas directory
    :param img_path: path to the target image
    :param mask_path: path to the target image mask
    :param save_dir: path to the directory where to save the results
    """

    atlas_name = os.path.split(atlas_dir)[1]

    atlas_img_path = os.path.join(atlas_dir, 'srr.nii.gz')
    atlas_seg_path = os.path.join(atlas_dir, 'parcellation.nii.gz')
    atlas_mask_path = os.path.join(atlas_dir, 'mask.nii.gz')

    save_dir_atlas = os.path.join(save_dir, "registered_atlases", atlas_name)

    # List of files created by reg_aladin, reg_f3d, reg_resample
    affine_path = os.path.join(save_dir_atlas, 'affine.txt')
    affine_warped_atlas_img_path = os.path.join(save_dir_atlas, 'affine_warped_atlas_img.nii.gz')
    cpp_path = os.path.join(save_dir_atlas, 'cpp.nii.gz')
    warped_atlas_img_path = os.path.join(save_dir_atlas, 'warped_atlas_img.nii.gz')
    warped_atlas_seg_path = os.path.join(save_dir_atlas, 'warped_atlas_seg.nii.gz')
    disp_field_path = os.path.join(save_dir_atlas, 'disp.nii.gz')
    lncc_distance_path = os.path.join(save_dir_atlas, 'lncc_distance.nii.gz')
    weights_path = os.path.join(save_dir_atlas, 'weights.nii.gz')

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

        if not os.path.exists(save_dir_atlas):
            os.makedirs(save_dir_atlas, exist_ok=True)

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

        # get the linearly interpolated warped atlas
        # the linear interpolation preserves the edges of the masked images better than cubic spline interpolation
        warped_atlas_linear_interp_nii = nib.load(warped_atlas_img_path.replace(".nii.gz", "_linear_interp.nii.gz"))
        warped_atlas_linear_interp = warped_atlas_linear_interp_nii.get_fdata(dtype=np.float32)

        # compute LNCC distance
        lncc_distance = get_lncc_distance(
            image=nib.load(img_path).get_fdata(dtype=np.float32),
            mask=nib.load(mask_path).get_fdata(dtype=np.float16).astype(np.uint8),
            atlas_warped_image=warped_atlas_linear_interp,
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
    for f_n in os.listdir(save_dir_atlas):
        p = os.path.join(save_dir_atlas, f_n)
        if p not in to_not_remove and not any([p.split(os.sep)[-1].startswith(s) for s in to_not_remove_which_starts_with]):
            os.system('rm %s' % p)

    return warped_atlas_seg_path, weights_path
