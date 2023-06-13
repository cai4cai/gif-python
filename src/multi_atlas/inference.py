import os
import time

import numpy as np
import nibabel as nib
from tqdm import tqdm

from src.multi_atlas.atlas_propagation import probabilistic_segmentation_prior
from src.multi_atlas.utils import compute_disp_from_cpp
from src.multi_atlas.multi_atlas_fusion_weights import log_heat_kernel_GIF

from multiprocessing import Pool

SUPPORTED_MERGING_METHOD = [
    'mean',
    'GIF',
]


def _weights_from_log_heat_kernels(log_heat_kernels):
    max_heat = log_heat_kernels.max(axis=0)
    x = log_heat_kernels - max_heat[None,:,:,:]
    exp_x = np.exp(x)
    norm = np.sum(exp_x, axis=0)
    w = exp_x / norm[None,:,:,:]
    return w


def nibabel_load_and_get_fdata(filepath):
    return nib.load(filepath).get_fdata()


def nibabel_load_and_get_fdata_and_weight(params):
    path = params[0]
    weights = params[1]
    return weights * nib.load(path).get_fdata()

def calculate_warped_prob_segmentation(param_list):
    folder, save_folder, reuse_existing_pred, force_recompute_heat_kernels, img_nii, mask_nii, num_class, grid_spacing, be, le, lp, only_affine = param_list
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
    to_not_remove = [  # paths to filter during the cleaning at the end
        expected_cpp_path,
        expected_aff_path,
        warped_altas_seg_onehot_path,
        warped_atlas_img_path,
        # expected_warped_atlas_path,
        # expected_disp_path,
        heat_kernel_path
    ]
    to_not_remove_which_starts_with = [
        "atlas_seg_",
        "warped_atlas_seg_",
    ]

    # Try to see what results we can reuse to save time
    compute_registration = True
    compute_heat_map = True
    if reuse_existing_pred:
        compute_registration = False
        compute_heat_map = False
        for p in to_not_remove:
            if not os.path.exists(p):
                compute_registration = True
                compute_heat_map = True
    if force_recompute_heat_kernels:
        compute_heat_map = True
        if not os.path.exists(warped_atlas_img_path) or not os.path.exists(disp_field_path):
            compute_registration = True


    time_0_reg = time.time()
    if not compute_registration:
        print('\n%s already exists.\nSkip registration.' % warped_altas_seg_onehot_path)
        # proba_atlas_prior_nii = nib.load(expected_output)
        # proba_atlas_prior = proba_atlas_prior_nii.get_fdata().astype(np.float32)
    else:
        template_nii = nib.load(os.path.join(folder, 'srr.nii.gz'))
        template_mask_nii = nib.load(os.path.join(folder, 'mask.nii.gz'))
        template_seg_nii = nib.load(os.path.join(folder, 'parcellation.nii.gz'))
        prob_warped_atlas_seg_l_paths, warped_atlas_seg_mask = probabilistic_segmentation_prior(
            image_nii=img_nii,
            mask_nii=mask_nii,
            template_nii=template_nii,
            template_seg_nii=template_seg_nii,
            template_mask_nii=template_mask_nii,
            atlas_seg_onehot_path=atlas_seg_onehot_path,
            warped_altas_seg_onehot_path=warped_altas_seg_onehot_path,
            warped_atlas_img_path=warped_atlas_img_path,
            num_class=num_class,
            grid_spacing=grid_spacing,
            be=be,
            le=le,
            lp=lp,
            save_folder_path=save_folder_atlas,
            affine_only=only_affine,
        )

    print(f"Registration and resampling of {atlas_name} completed after {time.time() - time_0_reg:.3f} seconds")

    # Compute the heat kernel for the GIF-like fusion
    time_0_heat = time.time()
    if not compute_heat_map:  # Load the existing heat map (skip computation)
        print('\n%s already exists.\nSkip heat kernel calculation.' % heat_kernel_path)
        # log_heat_kernel_nii = nib.load(expected_heat_kernel)
        # log_heat_kernel = log_heat_kernel_nii.get_fdata().astype(np.float32)
    else:  # Compute the heat map
        warped_atlas_nii = nib.load(warped_atlas_img_path)
        warped_atlas = warped_atlas_nii.get_fdata().astype(np.float32)

        if compute_registration:  # Recompute the displacement field if registration was run
            expected_cpp_path = os.path.join(save_folder_atlas, 'cpp.nii.gz')
            expected_img_path = os.path.join(save_folder_atlas, 'img.nii.gz')
            compute_disp_from_cpp(expected_cpp_path, expected_img_path, disp_field_path)

        deformation = nib.load(disp_field_path).get_fdata().astype(np.float32)
        log_heat_kernel, lssd, disp_norm = log_heat_kernel_GIF(
            image=img_nii.get_fdata().astype(np.float32),
            mask=mask_nii.get_fdata().astype(np.uint8),
            atlas_warped_image=warped_atlas,
            atlas_warped_mask=warped_atlas_seg_mask,
            deformation_field=deformation,
        )
        # Save the heat kernel
        log_heat_kernel_nii = nib.Nifti1Image(log_heat_kernel, warped_atlas_nii.affine)
        nib.save(log_heat_kernel_nii, heat_kernel_path)

    print(f"Heat kernel calculation of {atlas_name} completed after {time.time() - time_0_heat:.3f} seconds")

    # Cleaning - remove the files that we will not need anymore
    time_0_clean = time.time()
    for f_n in os.listdir(save_folder_atlas):
        p = os.path.join(save_folder_atlas, f_n)
        if not p in to_not_remove and not any([p.split(os.sep)[-1].startswith(s) for s in to_not_remove_which_starts_with]):
            os.system('rm %s' % p)
    print(f"Cleaning completed after {time.time() - time_0_clean:.3f} seconds")

    return prob_warped_atlas_seg_l_paths, heat_kernel_path

def multi_atlas_segmentation(img_nii,
                             mask_nii,
                             atlas_folder_list,
                             num_class,
                             grid_spacing,
                             be,
                             le,
                             lp,
                             save_folder,
                             only_affine,
                             merging_method='GIF',
                             reuse_existing_pred=False,
                             force_recompute_heat_kernels=False
                             ):

    '''
    Args:
        img_nii: image to segment
        mask_nii: registration mask
        atlas_folder_list:
    '''
    assert merging_method in SUPPORTED_MERGING_METHOD, \
        "Merging method %s not supported. Only %s supported." % (merging_method, str(SUPPORTED_MERGING_METHOD))
    time_0 = time.time()

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Register the atlas segmentations to the input image

    param_lists = [[folder,
                    save_folder,
                    reuse_existing_pred,
                    force_recompute_heat_kernels,
                    img_nii,
                    mask_nii,
                    num_class,
                    grid_spacing,
                    be,
                    le,
                    lp,
                    only_affine]
                   for folder in atlas_folder_list]

    num_pools = 2

    with Pool(num_pools) as p:
        print(f"Start multiprocessing with {num_pools} pools...")
        out =  p.map(calculate_warped_prob_segmentation, param_lists)

    proba_seg_path_list_a_l = [p[0] for p in out]
    log_heat_kernel_path_list = [p[1] for p in out]


    # Merging probabilities
    t_0_merge = time.time()
    MAX_NUM_VOX_IN_MEM = 8e9/4 # xGB/4bytes(float32)
    num_atlases = len(proba_seg_path_list_a_l)
    block_edge_len = int(np.ceil(MAX_NUM_VOX_IN_MEM**(1/3)/num_atlases))
    # Merge the proba predictions
    print("Merging probabilities...")

    if merging_method == 'GIF':
        print("Start weights calculation...")
        t_0_weight = time.time()
        with Pool(num_pools) as p:
            log_heat_kernels = p.map(nibabel_load_and_get_fdata, log_heat_kernel_path_list)  # n_atlas, n_x, n_y, n_z
        log_heat_kernels = np.stack(log_heat_kernels, axis=0)
        max_heat = log_heat_kernels.max(axis=0)
        x = log_heat_kernels - max_heat[None, :, :, :]
        exp_x = np.exp(x)
        norm = np.sum(exp_x, axis=0)
        weights = exp_x / norm[None, :, :, :]  # num_atlases x H x W x D

        print(f"Weights completed after {time.time() - t_0_weight:.3f} seconds")

        multi_atlas_proba_seg = np.zeros(tuple(weights.shape[1:]+(num_class,)))
        proba_seg_path_list_l_a = list(map(list, zip(*proba_seg_path_list_a_l)))  # transpose the list of lists

        print(f"Combining weights with probabilities...")
        t_0_combprobs = time.time()
        with Pool(num_pools) as p:
            for l, proba_seg_path_list_a in enumerate(proba_seg_path_list_l_a):
                print(l)

                weighted_proba_seg_a = p.map(nibabel_load_and_get_fdata_and_weight, [[proba_seg_path_list_a[a], weights[a, :, :, :]] for a in range(num_atlases)])

                print(f"Loading and weighting of atlas probabilities for class {l} completed...")

                weighted_proba_seg_a = np.array(weighted_proba_seg_a)  # num_atlas x H x W x D ; converts list of arrays to numpy array
                weighted_proba_seg = np.sum(weighted_proba_seg_a, axis=0)  # H x W x D; final weighted sum for class l

                multi_atlas_proba_seg[:, :, :, l] = weighted_proba_seg # H x W x D x num_classes

        print(f"Combining weights with probabilities completed after {time.time() - t_0_combprobs:.3f} seconds")

    else:  # Vanilla average
        multi_atlas_proba_seg = np.mean(np.stack([nib.load(f).get_fdata() for f in proba_seg_path_list_a_l], axis=0), axis=0)

    t_end = time.time()
    print(f"Merging completed after {t_end - t_0_merge:.3f} seconds")

    print(f"Multi-Atlas segmentation completed after {time.time() - time_0:.3f} seconds")
    return multi_atlas_proba_seg
