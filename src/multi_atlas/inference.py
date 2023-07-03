import cProfile
import os
import subprocess
import time

import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

from src.multi_atlas.atlas_propagation import probabilistic_segmentation_prior
from src.multi_atlas.utils import compute_disp_from_cpp
from src.multi_atlas.utils import structure_seg_from_tissue_seg
from src.multi_atlas.multi_atlas_fusion_weights import log_heat_kernel_GIF
from src.utils.definitions import RESAMPLE_METHOD, MULTIPROCESSING, NIFTYSEG_PATH

from ast import literal_eval

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

def select_label_from_labelmap_and_weight(params):
    array = params[0]
    weights = params[1]
    label = params[2]

    array_onehot_l = np.zeros_like(array)
    array_onehot_l[array==label] = 1
    return weights * array_onehot_l

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
        "warped_atlas_seg",
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
        warped_atlas_path_or_prob_seg_l_paths, warped_atlas_seg_mask = probabilistic_segmentation_prior(
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

    return warped_atlas_path_or_prob_seg_l_paths, heat_kernel_path

def multi_atlas_segmentation(img_path,
                             mask_path,
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

    """
    Multi-atlas segmentation using the GIF-like fusion method.
    :param img_path: Path to the input image
    :param mask_path: Path to the input mask
    :param atlas_folder_list: List of paths to the atlas folders
    :param num_class: Number of classes
    """
    assert merging_method in SUPPORTED_MERGING_METHOD, \
        "Merging method %s not supported. Only %s supported." % (merging_method, str(SUPPORTED_MERGING_METHOD))
    time_0 = time.time()

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    img_nii = nib.load(img_path)
    mask_nii = nib.load(mask_path)

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

    if MULTIPROCESSING:
        with Pool(num_pools) as p:
            print(f"Start multiprocessing with {num_pools} pools...")
            out =  p.map(calculate_warped_prob_segmentation, param_lists)
    else:
        out = []
        for params in param_lists:
            result = calculate_warped_prob_segmentation(params)
            out.append(result)

    warped_atlas_path_list_or_proba_seg_path_list_a_l = [p[0] for p in out]
    log_heat_kernel_path_list = [p[1] for p in out]

    # Merging probabilities
    t_0_merge = time.time()
    MAX_NUM_VOX_IN_MEM = 8e9/4 # xGB/4bytes(float32)
    num_atlases = len(warped_atlas_path_list_or_proba_seg_path_list_a_l)
    block_edge_len = int(np.ceil(MAX_NUM_VOX_IN_MEM**(1/3)/num_atlases))
    # Merge the proba predictions
    print("Merging probabilities...")

    if merging_method == 'GIF':
        print("Start weights calculation...")
        t_0_weight = time.time()

        if MULTIPROCESSING:
            with Pool(num_pools) as p:
                log_heat_kernels = p.map(nibabel_load_and_get_fdata, log_heat_kernel_path_list)  # n_atlas, n_x, n_y, n_z
        else:
            log_heat_kernels = []
            for path in log_heat_kernel_path_list:
                kernel_data = nibabel_load_and_get_fdata(path)
                log_heat_kernels.append(kernel_data)

        log_heat_kernels = np.stack(log_heat_kernels, axis=0)
        max_heat = log_heat_kernels.max(axis=0)
        x = log_heat_kernels - max_heat[None, :, :, :]
        exp_x = np.exp(x)
        norm = np.sum(exp_x, axis=0)
        weights = exp_x / norm[None, :, :, :]  # num_atlases x H x W x D

        print(f"Weights completed after {time.time() - t_0_weight:.3f} seconds")

############################################################################################################
        # Merge the proba predictions for tissue segmentation
        tissue_info_csv_path = os.path.join(os.path.dirname(atlas_folder_list[0]), 'tissues_info.csv')
        tissue_dict = pd.read_csv(tissue_info_csv_path, index_col="label").to_dict(orient='dict')

        structure_info_csv_path = os.path.join(os.path.dirname(atlas_folder_list[0]), 'structures_info.csv')
        structure_df = pd.read_csv(structure_info_csv_path, index_col="label", converters={'tissues': literal_eval})\

        structure_dict = structure_df.to_dict(orient='dict')

        num_tissue = len(tissue_dict['name'])
        multi_atlas_tissue_prior = np.zeros(tuple(weights.shape[1:] + (num_tissue,)))
        multi_atlas_proba_seg = np.zeros(tuple(weights.shape[1:] + (num_class,)), dtype=np.float32)

        print(f"Apply weights to labels...")
        t_0_combprobs = time.time()
        with Pool(num_pools) as p:
            if RESAMPLE_METHOD == 0:
                warped_atlases = np.array(p.map(nibabel_load_and_get_fdata,
                                                [warped_atlas_path_list_or_proba_seg_path_list_a_l[a] for a in
                                                 range(num_atlases)]))  # num_atlases x H x W x D
            elif RESAMPLE_METHOD == 1:
                proba_seg_path_list_l_a = list(
                    map(list, zip(*warped_atlas_path_list_or_proba_seg_path_list_a_l)))  # transpose the list of lists

            for l in tqdm(range(num_class)):
                if not l in structure_dict['tissues']:
                    print(f"Label {l} is not present in the structures_info.csv. Assign zero probability...")
                    continue

                if RESAMPLE_METHOD == 0:
                    weighted_proba_seg_a = p.map(select_label_from_labelmap_and_weight,
                                                 [[warped_atlases[a], weights[a, :, :, :], l] for a in
                                                  range(num_atlases)])

                elif RESAMPLE_METHOD == 1:
                    proba_seg_path_list_a = proba_seg_path_list_l_a[l]
                    weighted_proba_seg_a = p.map(nibabel_load_and_get_fdata_and_weight,
                                                 [[proba_seg_path_list_a[a], weights[a, :, :, :]] for a in
                                                  range(num_atlases)])

                weighted_proba_seg_a = np.array(weighted_proba_seg_a)  # num_atlas x H x W x D ; converts list of arrays to numpy array

                # sum all probabilities that are part of the same tissue, while dividing probabilities by the number of tissues they are assigned to
                weighted_proba_seg = np.sum(weighted_proba_seg_a, axis=0)  # H x W x D; final weighted sum for class l


                assigned_tissues = structure_dict['tissues'][l]
                for t in assigned_tissues:
                    multi_atlas_tissue_prior[:, :, :, t] += weighted_proba_seg / len(assigned_tissues) # H x W x D x num_tissue

                multi_atlas_proba_seg[:, :, :, l] = weighted_proba_seg  # H x W x D x num_classes

        # Convert to tissue labelmap and save
        multi_atlas_tissue_prior_nii = nib.Nifti1Image(multi_atlas_tissue_prior, affine=img_nii.affine)
        nib.save(multi_atlas_tissue_prior_nii, os.path.join(save_folder, f"multi_atlas_tissue_prior.nii.gz"))

        # run seg_EM algorithm to get final tissue segmentation
        # Define file paths
        seg_EM_input_filename = img_path
        seg_EM_output_filename = os.path.join(save_folder, f"multi_atlas_tissue_seg.nii.gz")
        seg_EM_mask_filename = mask_path
        seg_EM_prior_filename = os.path.join(save_folder, f"multi_atlas_tissue_prior.nii.gz")
        seg_EM_n_priors = 1

        # Define other input options
        seg_EM_verbose_level = 0
        seg_EM_max_iterations = 30
        seg_EM_min_iterations = 3
        seg_EM_bias_field_order = 5
        seg_EM_bias_field_thresh = 0.05
        seg_EM_mrf_beta = 0.1

        command = [os.path.join(NIFTYSEG_PATH, 'seg_EM'),
                    '-in', seg_EM_input_filename,
                    '-out', seg_EM_output_filename,
                    '-mask', seg_EM_mask_filename,
                    '-priors4D', seg_EM_prior_filename,
                    '-v', str(seg_EM_verbose_level),
                    '-max_iter', str(seg_EM_max_iterations),
                    '-min_iter', str(seg_EM_min_iterations),
                    '-bc_order', str(seg_EM_bias_field_order),
                    '-bc_thresh', str(seg_EM_bias_field_thresh),
                    '-mrf_beta', str(seg_EM_mrf_beta)]

        # Run the command
        subprocess.call(command)

        multi_atlas_tissue_seg = np.argmax(nib.load(seg_EM_output_filename).get_fdata(), axis=-1).astype(np.uint8)

        # get the label with the maximum probability according to multi_atlas_proba_seg under the condition that one of
        # the assigned tissues corresponds to the tissue in multi_atlas_tissue_seg
        final_parcellation = structure_seg_from_tissue_seg(multi_atlas_tissue_seg, multi_atlas_proba_seg, structure_dict['tissues'])

        print(f"\nCombining weights with probabilities completed after {time.time() - t_0_combprobs:.3f} seconds")

        predicted_segmentation_nii = nib.Nifti1Image(final_parcellation, img_nii.affine)
        nib.save(predicted_segmentation_nii, os.path.join(save_folder, "predicted_segmentation.nii.gz"))

        return final_parcellation