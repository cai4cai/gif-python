import os
import time

import numpy as np
import nibabel as nib
import pandas as pd
from ast import literal_eval

from src.multi_atlas.atlas_registration_and_resampling import warp_atlas_and_calc_similarity_weights
from src.multi_atlas.multi_atlas_fusion import get_multi_atlas_proba_seg, get_multi_atlas_tissue_prior, get_structure_seg_from_tissue_seg
from src.multi_atlas.utils import seg_EM, nibabel_load_and_get_fdata

from src.utils.definitions import MULTIPROCESSING, NUM_POOLS

if MULTIPROCESSING:
    from multiprocessing import Pool


def multi_atlas_segmentation(img_path,
                             mask_path,
                             atlas_dir_list,
                             structure_info_csv_path,
                             tissue_info_csv_path,
                             save_dir,
                             ):

    """
    Multi-atlas segmentation using the GIF-like fusion method.
    :param img_path: Path to the input image
    :param mask_path: Path to the input mask
    :param atlas_dir_list: List of paths to the atlas dirs
    :param structure_info_csv_path: Path to the structure info csv file that contains the label, the name and a list of
        the tissues that the structure can be part of, e.g.: 8,Right Accumbens Area,"[3, 4]"
    :param tissue_info_csv_path: Path to the tissue info csv file that contains the tissue class number, and the tissue
        name, e.g.: 3, White Matter
    :param save_dir: Path to the directory where the results will be saved
    """

    # define the paths to the files that will be created
    multi_atlas_proba_seg_path = os.path.join(save_dir, f"multi_atlas_proba_seg.nii.gz")
    multi_atlas_tissue_prior_path = os.path.join(save_dir, f"multi_atlas_tissue_prior.nii.gz")
    multi_atlas_tissue_seg_path = os.path.join(save_dir, f"multi_atlas_tissue_seg.nii.gz")
    final_parcel_path = os.path.join(save_dir, f"final_parcellation.nii.gz")

    time_0 = time.time()

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    img_affine = nib.load(img_path).affine
    mask_nii = nib.load(mask_path)
    img_mask = mask_nii.get_fdata(dtype=np.float16).astype(np.uint8)

    ####################################################################################################################
    # Register the atlas segmentations to the input image and compute the similarity weights
    ####################################################################################################################
    
    param_tuples = [(atlas_dir, save_dir, img_path, mask_path,) for atlas_dir in atlas_dir_list]

    if MULTIPROCESSING:
        with Pool(NUM_POOLS) as p:
            print(f"Start multiprocessing with {NUM_POOLS} pools...")
            out = p.starmap(warp_atlas_and_calc_similarity_weights, param_tuples)
    else:
        out = []
        for params in param_tuples:
            result = warp_atlas_and_calc_similarity_weights(*params)
            out.append(result)

    warped_atlas_path_list = [p[0] for p in out]
    weights_path_list = [p[1] for p in out]

    print(f"Registration and similarity weights calculation completed after {time.time() - time_0:.3f} seconds")

    ####################################################################################################################
    # Merging probabilities
    ####################################################################################################################
    # Weight and merge the atlas segmentations to get the multi-atlas probabilities for each structure
    print("Merging probabilities...")
    num_atlases = len(warped_atlas_path_list)

    # Load the weights and the atlas segmentations
    print("Load weights from disk...")
    t_0_weight = time.time()

    if MULTIPROCESSING:
        with Pool(NUM_POOLS) as p:
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

    ####################################################################################################################
    # read structure and tissue info
    ####################################################################################################################

    tissue_dict = pd.read_csv(tissue_info_csv_path, index_col="label").to_dict(orient='dict')
    num_tissue = len(tissue_dict['name'])

    structure_df = pd.read_csv(structure_info_csv_path, index_col="label", converters={'tissues': literal_eval})
    structure_dict = structure_df.to_dict(orient='dict')

    # length of label dimension (assuming the labels are consecutive)
    num_class = np.max(list(structure_dict['name'].keys())) + 1

    print(f"Load the warped atlas segmentations from disk ...")
    t_0_load_warped_atlases = time.time()
    with Pool(NUM_POOLS) as p:
        warped_atlases = np.array(p.starmap(nibabel_load_and_get_fdata, [(warped_atlas_path_list[a], np.uint8) for a in
                                            range(num_atlases)]))  # num_atlases x H x W x D

    print(f"Warped atlas segmentations loading completed after {time.time() - t_0_load_warped_atlases:.3f} seconds")

    #################################################################################################################### 
    # calculate the multi-atlas probabilities for each structure  # H x W x D x num_class
    ####################################################################################################################
    
    multi_atlas_proba_seg = np.zeros((*warped_atlases.shape[1:], num_class), dtype=np.float32)
    multi_atlas_proba_seg = get_multi_atlas_proba_seg(warped_atlases,
                                                      weights,
                                                      np.array(list(structure_dict['name'].keys())),
                                                      multi_atlas_proba_seg)

    # # save the merged probabilities (time consuming, since H x W x D x num_class is large)
    # multi_atlas_proba_seg_nii = nib.Nifti1Image(multi_atlas_proba_seg, affine=img_affine)
    # nib.save(multi_atlas_proba_seg_nii, multi_atlas_proba_seg_path)

    ####################################################################################################################
    # calculate the multi-atlas tissue prior by distributing the probabilities of each structure to its assigned tissues
    ####################################################################################################################
    
    time_0_tissue_prior = time.time()

    multi_atlas_tissue_prior = get_multi_atlas_tissue_prior(multi_atlas_proba_seg,
                                                            structure_dict,
                                                            num_class,
                                                            num_tissue,
                                                            img_mask,
                                                            )

    # save tissue prior
    print(f"Saving multi-atlas tissue prior...")
    multi_atlas_tissue_prior_nii = nib.Nifti1Image(multi_atlas_tissue_prior, affine=img_affine)
    nib.save(multi_atlas_tissue_prior_nii, multi_atlas_tissue_prior_path)

    print(f"Multi-atlas tissue prior calculation completed after {time.time() - time_0_tissue_prior:.3f} seconds")

    ####################################################################################################################
    # run seg_EM algorithm to get final tissue segmentation
    ####################################################################################################################
    
    print(f"Running seg_EM algorithm...")
    t_0_segEM = time.time()
    seg_EM_params = {'input_filename': img_path,
                     'output_filename': multi_atlas_tissue_seg_path,
                     'mask_filename': mask_path,
                     'prior_filename': multi_atlas_tissue_prior_path, 
                     'verbose_level': 0,
                     'max_iterations': 30, 
                     'min_iterations': 3, 
                     'bias_field_order': 4, 
                     'bias_field_thresh': 0.05,
                     'mrf_beta': 0.1}

    # run seg_EM algorithm
    seg_EM(**seg_EM_params)

    # load the result
    print(f"Loading seg_EM result...")
    multi_atlas_tissue_seg = np.argmax(
        nib.load(seg_EM_params['output_filename']).get_fdata(dtype=np.float16), axis=-1).astype(np.uint8)

    print(f"Running seg_EM algorithm completed after {time.time() - t_0_segEM:.3f} seconds")

    ####################################################################################################################
    # get the final structure segmentation
    ####################################################################################################################
    
    # get the label with the maximum probability according to multi_atlas_proba_seg under the condition that one of
    # the assigned tissues corresponds to the tissue in multi_atlas_tissue_seg
    print(f"Running structure segmentation...")
    t_0_struct_seg = time.time()
    final_parcellation = get_structure_seg_from_tissue_seg(multi_atlas_tissue_seg, multi_atlas_proba_seg, structure_dict['tissues'])
    print(f"Running structure segmentation completed after {time.time() - t_0_struct_seg:.3f} seconds")

    # save the final parcellation
    predicted_segmentation_nii = nib.Nifti1Image(final_parcellation, img_affine)
    nib.save(predicted_segmentation_nii, final_parcel_path)

    return final_parcellation
