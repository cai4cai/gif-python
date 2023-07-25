import os
import time
import nibabel as nib
import numpy as np

from src.multi_atlas.atlas_propagation import register_and_resample_atlas
from src.multi_atlas.utils import compute_disp_from_cpp
from src.multi_atlas.multi_atlas_fusion_weights import get_lncc_distance

def warp_atlas_and_calc_similarity_weights( folder,
                                            save_folder,
                                            img_nii,
                                            mask_nii,
                                            num_class,
                                            only_affine):

    atlas_name = os.path.split(folder)[1]
    save_folder_atlas = os.path.join(save_folder, atlas_name)

    # List of files that should exist at the end of the segmentation
    expected_cpp_path = os.path.join(save_folder_atlas, 'cpp.nii.gz')
    expected_aff_path = os.path.join(save_folder_atlas, 'outputAffine.txt')
    atlas_seg_onehot_path = os.path.join(save_folder_atlas, 'atlas_seg_onehot.nii.gz')
    warped_altas_seg_onehot_path = os.path.join(save_folder_atlas, 'warped_atlas_seg_onehot.nii.gz')
    warped_atlas_img_path = os.path.join(save_folder_atlas, 'warped_atlas_img.nii.gz')
    disp_field_path = os.path.join(save_folder_atlas, 'disp.nii.gz')
    heat_kernel_path = os.path.join(save_folder_atlas, 'log_heat_kernel.nii.gz')
    lncc_distance_path = os.path.join(save_folder_atlas, 'lncc_distance.nii.gz')
    weights_path = os.path.join(save_folder_atlas, 'weights.nii.gz')

    # paths to NOT delete during the cleaning at the end
    to_not_remove = [
        expected_cpp_path,
        expected_aff_path,
        warped_altas_seg_onehot_path,
        warped_atlas_img_path,
        heat_kernel_path,
        lncc_distance_path,
        disp_field_path,
        weights_path

    ]
    to_not_remove_which_starts_with = [
        "atlas_seg_",
        "warped_atlas_seg",
        "warped_atlas_seg_",
        "warped_atlas_img_linear_interp",
    ]


    time_0_reg = time.time()

    template_nii = nib.load(os.path.join(folder, 'srr.nii.gz'))
    template_mask_nii = nib.load(os.path.join(folder, 'mask.nii.gz'))
    template_seg_nii = nib.load(os.path.join(folder, 'parcellation.nii.gz'))

    # Compute the warped atlas (image and segmentation)
    warped_atlas_path, warped_atlas_seg_mask = register_and_resample_atlas(
                                                image_nii=img_nii,
                                                mask_nii=mask_nii,
                                                template_nii=template_nii,
                                                template_seg_nii=template_seg_nii,
                                                template_mask_nii=template_mask_nii,
                                                atlas_seg_onehot_path=atlas_seg_onehot_path,
                                                warped_altas_seg_onehot_path=warped_altas_seg_onehot_path,
                                                warped_atlas_img_path=warped_atlas_img_path,
                                                num_class=num_class,
                                                save_folder_path=save_folder_atlas,
                                                affine_only=only_affine,
                                            )

    print(f"Registration and resampling of {atlas_name} completed after {time.time() - time_0_reg:.3f} seconds")

    # Compute the LNCC distance for the GIF-like fusion
    time_0_heat = time.time()

    warped_atlas_nii = nib.load(warped_atlas_img_path)

    expected_cpp_path = os.path.join(save_folder_atlas, 'cpp.nii.gz')
    expected_img_path = os.path.join(save_folder_atlas, 'img.nii.gz')
    compute_disp_from_cpp(expected_cpp_path, expected_img_path, disp_field_path)

    # get the linearly interpolated warped atlas
    warped_atlas_linear_interp_nii = nib.load(warped_atlas_img_path.replace(".nii.gz", "_linear_interp.nii.gz"))
    warped_atlas_linear_interp = warped_atlas_linear_interp_nii.get_fdata(dtype=np.float32)

    # compute LNCC distance
    lncc_distance = get_lncc_distance(
        image=img_nii.get_fdata(dtype=np.float32),
        mask=mask_nii.get_fdata(dtype=np.float16).astype(np.uint8),
        atlas_warped_image=warped_atlas_linear_interp,
        save_folder_path=save_folder_atlas,
        affine = warped_atlas_nii.affine,
        spacing=warped_atlas_nii.header.get_zooms()[0:3],
    )

    # save the lncc distance
    lncc_distance_nii = nib.Nifti1Image(lncc_distance, warped_atlas_nii.affine)
    nib.save(lncc_distance_nii, lncc_distance_path)

    # calculate weights
    temperature = 0.15
    weights = np.exp(-lncc_distance / temperature)

    # save the weights
    weights_nii = nib.Nifti1Image(weights, warped_atlas_nii.affine)
    nib.save(weights_nii, weights_path)

    print(f"Heat kernel calculation of {atlas_name} completed after {time.time() - time_0_heat:.3f} seconds")

    # Cleaning - remove the files that we will not need anymore
    time_0_clean = time.time()
    for f_n in os.listdir(save_folder_atlas):
        p = os.path.join(save_folder_atlas, f_n)
        if not p in to_not_remove and not any([p.split(os.sep)[-1].startswith(s) for s in to_not_remove_which_starts_with]):
            os.system('rm %s' % p)
    print(f"Cleaning completed after {time.time() - time_0_clean:.3f} seconds")

    return warped_atlas_path, weights_path