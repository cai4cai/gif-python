import multiprocessing
import os
import time

import numpy as np
import nibabel as nib
import pandas as pd
from numba import njit
from tqdm import tqdm

from src.multi_atlas.atlas_propagation import probabilistic_segmentation_prior
from src.multi_atlas.utils import compute_disp_from_cpp, structure_seg_from_tissue_seg, seg_EM
from src.multi_atlas.multi_atlas_fusion_weights import log_heat_kernel_GIF, get_lncc_distance
from src.utils.definitions import RESAMPLE_METHOD, MULTIPROCESSING

from ast import literal_eval

from multiprocessing import Pool

SUPPORTED_MERGING_METHOD = [
    'GIF',
]

# global dictionary to store variables that are shared across processes
var_dict = {}

def init_worker(warped_atlases2, warped_atlases_shape, weights2, weights_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['warped_atlases'] = warped_atlases2
    var_dict['warped_atlases_shape'] = warped_atlases_shape
    var_dict['weights'] = weights2
    var_dict['weights_shape'] = weights_shape


def _weights_from_log_heat_kernels(log_heat_kernels):
    max_heat = log_heat_kernels.max(axis=0)
    x = log_heat_kernels - max_heat[None,:,:,:]
    exp_x = np.exp(x)
    norm = np.sum(exp_x, axis=0)
    w = exp_x / norm[None,:,:,:]
    return w


def merge_label_weights_multiprocessing(params):
    print(f"merge_label_weights for label {params[0]}")
    l = params[0]
    structure_dict = params[1]

    weights = var_dict['weights']
    warped_atlases = var_dict['warped_atlases']

    if not l in structure_dict['tissues']:
        print(f"Label {l} is not present in the structures_info.csv. Assign zero probability...")

        # print shape of returned array
        print(f"Shape of returned array: {np.zeros(var_dict['warped_atlases_shape'][1:]).shape}")

        return np.zeros(var_dict['warped_atlases_shape'][1:])

    # return summed weights for the current label for all atlases
    num_atlases = var_dict['warped_atlases_shape'][0]
    num_voxels = np.prod(var_dict['warped_atlases_shape'][1:])
    labels_weights = np.zeros(num_voxels)

    try:
        for a in range(num_atlases):
            for i in range(num_voxels):
                # get the linear index of the current voxel of atlas a
                idx = i + a * num_voxels
                # if the current voxel of atlas a has label l and a weight > 0
                if weights[idx]>0 and warped_atlases[idx] == l:
                    # add the weight of the current voxel of atlas a to the
                    # corresponding voxel in labels_weights
                    labels_weights[i] += weights[idx]
    except:
        print(f"Error in merge_label_weights for label {l}, atlas {a}, voxel {i}")
        print(f" Length of warped_atlases: {len(warped_atlases)}")
        print(f" Length of weights: {len(weights)}")
        raise Exception("Error in merge_label_weights")

    labels_weights = labels_weights.reshape(*var_dict['warped_atlases_shape'][1:])

    return labels_weights

def merge_label_weights(label, structure_dict, warped_atlases, weights):
    l = label
    if not l in structure_dict['tissues']:
        print(f"Label {l} is not present in the structures_info.csv. Assign zero probability...")
        return np.zeros(warped_atlases.shape[1:], dtype=np.float32)

    # return summed weights for the current label for all atlases
    labels_weights = weights * (warped_atlases == l)

    # sum over all atlases
    labels_weights = np.sum(labels_weights, axis=0)

    return labels_weights

def nibabel_load_and_get_fdata(filepath):
    return nib.load(filepath).get_fdata()

def nibabel_load_and_get_fdata_as_uint8(filepath):
    print(f"Loading {filepath}")
    out = nib.load(filepath).get_fdata().astype(np.uint8)
    print(f"Loaded {filepath}")
    return out

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
    lncc_distance_path = os.path.join(save_folder_atlas, 'lncc_distance.nii.gz')
    weights_path = os.path.join(save_folder_atlas, 'weights.nii.gz')
    to_not_remove = [  # paths to filter during the cleaning at the end
        expected_cpp_path,
        expected_aff_path,
        warped_altas_seg_onehot_path,
        warped_atlas_img_path,
        # expected_warped_atlas_path,
        # expected_disp_path,
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
        # log_heat_kernel, lssd, disp_norm = log_heat_kernel_GIF(
        #     image=img_nii.get_fdata().astype(np.float32),
        #     mask=mask_nii.get_fdata().astype(np.uint8),
        #     atlas_warped_image=warped_atlas,
        #     atlas_warped_mask=warped_atlas_seg_mask,
        #     deformation_field=deformation,
        # )
        # # Save the heat kernel
        # log_heat_kernel_nii = nib.Nifti1Image(log_heat_kernel, warped_atlas_nii.affine)
        # nib.save(log_heat_kernel_nii, heat_kernel_path)

        # get the linearly interpolated warped atlas
        warped_atlas_linear_interp_nii = nib.load(warped_atlas_img_path.replace(".nii.gz", "_linear_interp.nii.gz"))
        warped_atlas_linear_interp = warped_atlas_linear_interp_nii.get_fdata().astype(np.float32)

        ## aaron: LNCC distance
        lncc_distance = get_lncc_distance(
            image=img_nii.get_fdata().astype(np.float32),
            mask=mask_nii.get_fdata().astype(np.uint8),
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

    return warped_atlas_path_or_prob_seg_l_paths, weights_path

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

    num_pools = 14

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
    weights_path_list = [p[1] for p in out]

    # Merging probabilities
    num_atlases = len(warped_atlas_path_list_or_proba_seg_path_list_a_l)
    # Merge the proba predictions
    print("Merging probabilities...")


    print("Start weights loading from disk...")
    t_0_weight = time.time()

    if MULTIPROCESSING:
        with Pool(num_pools) as p:
            weights_list = p.map(nibabel_load_and_get_fdata, weights_path_list)  # n_atlas * [H, W, D]
    else:
        weights_list = []
        for path in weights_path_list:
            weight = nibabel_load_and_get_fdata(path)
            weights_list.append(weight)

    # calculate the weights from the heat kernels
    weights = np.stack(weights_list, axis=0)  # n_atlas, H, W, D
    # set NaN in the weights to 0
    weights[np.isnan(weights)] = 0

    print(f"Weights loading completed after {time.time() - t_0_weight:.3f} seconds")


    # Merge the proba predictions for structure segmentation

    # get structure info
    tissue_info_csv_path = os.path.join(os.path.dirname(atlas_folder_list[0]), 'tissues_info.csv')
    tissue_dict = pd.read_csv(tissue_info_csv_path, index_col="label").to_dict(orient='dict')

    structure_info_csv_path = os.path.join(os.path.dirname(atlas_folder_list[0]), 'structures_info.csv')
    structure_df = pd.read_csv(structure_info_csv_path, index_col="label", converters={'tissues': literal_eval})\

    structure_dict = structure_df.to_dict(orient='dict')

    # create 2D numpy array that indicates which structure is present in which tissue
    num_tissue = len(tissue_dict['name'])
    struc_in_tiss_mat = np.zeros((num_class, num_tissue), dtype=np.bool)
    for t in range(num_tissue):
        for l in structure_dict['tissues']:
            struc_in_tiss_mat[l, t] = True if t in structure_dict['tissues'][l] else False

    print(f"Merge label weights from all atlases...")
    t_0_combprobs = time.time()
    with Pool(num_pools) as p:
        if RESAMPLE_METHOD == 0:
            warped_atlases = np.array(p.map(nibabel_load_and_get_fdata_as_uint8,
                                            [warped_atlas_path_list_or_proba_seg_path_list_a_l[a] for a in
                                             range(num_atlases)]))  # num_atlases x H x W x D
        elif RESAMPLE_METHOD == 1:
            proba_seg_path_list_l_a = list(
                map(list, zip(*warped_atlas_path_list_or_proba_seg_path_list_a_l)))  # transpose the list of lists

########################################################################################################################
    # @njit
    # def get_label_weigths(warped_atlases, weights, label):
    #     return np.sum(weights * (warped_atlases == label), axis=0)

    @njit
    def get_multi_atlas_proba_seg(warped_atlases, weights, labels_array):
        multi_atlas_proba_seg = np.zeros((*warped_atlases.shape[1:], num_class), dtype=np.float32)

        for x in range(warped_atlases.shape[1]):
            print(x)
            for y in range(warped_atlases.shape[2]):
                for z in range(warped_atlases.shape[3]):
                    for l in labels_array:
                        #print(l)
                        for a in range(warped_atlases.shape[0]):

                            if warped_atlases[a, x, y, z] == l:
                                multi_atlas_proba_seg[x, y, z, l] += weights[a, x, y, z]
                    # where ever the sum of probabilities is 0, set the first class (background) to 1
                    if np.sum(multi_atlas_proba_seg[x, y, z, :]) == 0:
                        multi_atlas_proba_seg[x, y, z, 0] = 1

        # # where ever the sum of probabilities is 0, set the first class (background) to 1
        # multi_atlas_proba_seg[np.sum(multi_atlas_proba_seg, axis=-1) == 0, 0] = 1

        return multi_atlas_proba_seg

    multi_atlas_proba_seg = get_multi_atlas_proba_seg(warped_atlases, weights, labels_array=np.array(list(structure_dict['name'].keys())))

    # # save the merged probabilities
    # multi_atlas_proba_seg_nii = nib.Nifti1Image(multi_atlas_proba_seg, affine=img_nii.affine)
    # nib.save(multi_atlas_proba_seg_nii, os.path.join(save_folder, f"multi_atlas_proba_seg.nii.gz"))

########################################################################################################################
    # Merge the proba predictions for tissue segmentation

    # @njit
    # def get_struc_label_prob(multi_atlas_proba_seg, nb_assigned_tissues):
    #     struc_label_prob = np.zeros(multi_atlas_proba_seg.shape[:3], dtype=np.float32)
    #     for x in range(multi_atlas_proba_seg.shape[0]):
    #         #print("x: ", x)
    #         for y in range(multi_atlas_proba_seg.shape[1]):
    #             #print("y: ", y)
    #             for z in range(multi_atlas_proba_seg.shape[2]):
    #                 #print("z: ", z)
    #                 struc_label_prob[x, y, z] = multi_atlas_proba_seg[x, y, z] / nb_assigned_tissues # H x W x D x num_tissue)
    #
    #     return struc_label_prob
    #
    #
    # @njit
    # def get_multi_atlas_tissue_prior(multi_atlas_proba_seg, struc_in_tiss_mat):
    #     multi_atlas_tissue_prior = np.zeros((*multi_atlas_proba_seg.shape[:3], num_tissue), dtype=np.float32)
    #
    #     print("multi_atlas_proba_seg.shape: ", multi_atlas_proba_seg.shape)
    #     print("struc_in_tiss_mat.shape: ", struc_in_tiss_mat.shape)
    #     print("multi_atlas_tissue_prior.shape: ", multi_atlas_tissue_prior.shape)
    #
    #     nb_assigned_tissues = np.sum(struc_in_tiss_mat, axis=1)  # num_class                nb_assigned_tissues = np.sum(struc_in_tiss_mat, axis=1)
    #
    #     for l in range(multi_atlas_proba_seg.shape[3]):
    #         print("label: ", l)
    #         for t in range(struc_in_tiss_mat.shape[1]):
    #             if struc_in_tiss_mat[l, t]:
    #                 #print("tissue: ", t)
    #                 multi_atlas_tissue_prior[:, :, :, t] += get_struc_label_prob(multi_atlas_proba_seg[:, :, :, l], nb_assigned_tissues[l])
    #
    #     return multi_atlas_tissue_prior
    #
    # multi_atlas_tissue_prior = get_multi_atlas_tissue_prior(multi_atlas_proba_seg, struc_in_tiss_mat)


    multi_atlas_tissue_prior = np.zeros(tuple(img_nii.shape + (num_tissue,)), dtype=np.float32)
    for l in range(num_class):
        if not l in structure_dict['tissues']:
            continue
        assigned_tissues = structure_dict['tissues'][l]
        for t in assigned_tissues:
            multi_atlas_tissue_prior[:, :, :, t] += multi_atlas_proba_seg[:, :, :, l] / len(assigned_tissues) # H x W x D x num_tissue

########################################################################################################################

    # # set all NaN to 0
    # multi_atlas_tissue_prior[np.isnan(multi_atlas_tissue_prior)] = 0

    # set the background tissue class to the inverse of the mask
    multi_atlas_tissue_prior[:, :, :, 0] = 1-mask_nii.get_fdata().astype(np.uint8)

    # set voxels that are not assigned to any tissue to the background tissue
    multi_atlas_tissue_prior[np.sum(multi_atlas_tissue_prior, axis=-1) == 0, 0] = 1

    # normalize the tissue prior
    multi_atlas_tissue_prior = multi_atlas_tissue_prior / np.sum(multi_atlas_tissue_prior, axis=-1, keepdims=True)

    # set everything below 0.01 to 0 and renormalise
    multi_atlas_tissue_prior[multi_atlas_tissue_prior < 0.01] = 0

    # set voxels that are not assigned to any tissue to the background tissue
    multi_atlas_tissue_prior[np.sum(multi_atlas_tissue_prior, axis=-1) == 0, 0] = 1

    # renormalise
    multi_atlas_tissue_prior = multi_atlas_tissue_prior / np.sum(multi_atlas_tissue_prior, axis=-1, keepdims=True)

    # save tissue prior
    multi_atlas_tissue_prior_nii = nib.Nifti1Image(multi_atlas_tissue_prior, affine=img_nii.affine)
    nib.save(multi_atlas_tissue_prior_nii, os.path.join(save_folder, f"multi_atlas_tissue_prior.nii.gz"))

    try:
        print(f"Combining weights with probabilities completed after {time.time() - t_0_combprobs:.3f} seconds")
    except:
        pass

    # run seg_EM algorithm to get final tissue segmentation
    print(f"Running seg_EM algorithm...")
    t_0_segEM = time.time()
    seg_EM_params = {}
    seg_EM_params['input_filename'] = img_path
    seg_EM_params['output_filename'] = os.path.join(save_folder, f"multi_atlas_tissue_seg.nii.gz")
    seg_EM_params['mask_filename'] = mask_path
    seg_EM_params['prior_filename'] = os.path.join(save_folder, f"multi_atlas_tissue_prior.nii.gz")
    seg_EM_params['verbose_level'] = 0
    seg_EM_params['max_iterations'] = 30
    seg_EM_params['min_iterations'] = 3
    seg_EM_params['bias_field_order'] = 4
    seg_EM_params['bias_field_thresh'] = 0.05
    seg_EM_params['mrf_beta'] = 0.1

    # run seg_EM algorithm
    seg_EM(**seg_EM_params)

    # load the result
    multi_atlas_tissue_seg = np.argmax(nib.load(seg_EM_params['output_filename']).get_fdata(), axis=-1).astype(np.uint8)

    print(f"Running seg_EM algorithm completed after {time.time() - t_0_segEM:.3f} seconds")

    # get the label with the maximum probability according to multi_atlas_proba_seg under the condition that one of
    # the assigned tissues corresponds to the tissue in multi_atlas_tissue_seg


    print(f"Running structure segmentation...")
    t_0_struct_seg = time.time()
    final_parcellation = structure_seg_from_tissue_seg(multi_atlas_tissue_seg, np.argsort(multi_atlas_proba_seg, axis=-1), struc_in_tiss_mat)
    print(f"Running structure segmentation completed after {time.time() - t_0_struct_seg:.3f} seconds")

    predicted_segmentation_nii = nib.Nifti1Image(final_parcellation, img_nii.affine)
    nib.save(predicted_segmentation_nii, os.path.join(save_folder, "predicted_segmentation.nii.gz"))

    return final_parcellation