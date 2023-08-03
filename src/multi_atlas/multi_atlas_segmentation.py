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

nib.Nifti1Header.quaternion_threshold = -np.finfo(np.float32).eps * 10  # CANDI dataset doesn't pass this check


def multi_atlas_segmentation(img_path,
                             mask_path,
                             atlas_paths_dicts_list,
                             structure_info_csv_path,
                             tissue_info_csv_path,
                             save_dir,
                             ):

    """
    Multi-atlas segmentation using the GIF-like fusion method.
    :param img_path: Path to the input image
    :param mask_path: (optional) Path to the input mask
    :param atlas_paths_dicts_list: List of dictionaries, each dictionary contains the name of the atlas, the path to
        the atlas image, the path to the atlas segmentation, and (optionally) the path to the atlas mask, e.g.:
        {"name": "atlas1", "img_path": "path/to/atlas1.nii.gz", "seg_path": "path/to/atlas1_seg.nii.gz",
        "mask_path": "path/to/atlas1_mask.nii.gz"}
    :param structure_info_csv_path: Path to the structure info csv file that contains the label, the name and a list of
        the tissues that the structure can be part of, e.g.: 8,Right Accumbens Area,"[3, 4]"
    :param tissue_info_csv_path: Path to the tissue info csv file that contains the tissue class number, and the tissue
        name, e.g.: 3, White Matter
    :param save_dir: Path to the directory where the results will be saved
    """

    ####################################################################################################################
    # Check inputs
    ####################################################################################################################
    assert os.path.exists(img_path), f"Input image does not exist: {img_path}"
    if mask_path is not None:
        assert os.path.exists(mask_path), f"Input mask does not exist: {mask_path}"
    else:
        mask_path = None
        print("No input mask for the target image was provided...")

    for atlas_dict in atlas_paths_dicts_list:
        assert os.path.exists(atlas_dict["img_path"]), f"Atlas image does not exist: {atlas_dict['img_path']}"
        assert os.path.exists(atlas_dict["seg_path"]), f"Atlas segmentation does not exist: {atlas_dict['seg_path']}"
        if "mask_path" in atlas_dict:
            assert os.path.exists(atlas_dict["mask_path"]), f"Atlas mask does not exist: {atlas_dict['mask_path']}"
        else:
            atlas_dict["mask_path"] = None
            print(f"No input mask for atlas {atlas_dict['name']} was provided...")

    assert os.path.exists(structure_info_csv_path), f"Structure info csv file does not exist: {structure_info_csv_path}"
    assert os.path.exists(tissue_info_csv_path), f"Tissue info csv file does not exist: {tissue_info_csv_path}"

    ####################################################################################################################
    # Prepare output paths
    ####################################################################################################################
    # define the paths to the files that will be created
    multi_atlas_proba_seg_path = os.path.join(save_dir, f"multi_atlas_proba_seg.nii.gz")
    multi_atlas_tissue_prior_path = os.path.join(save_dir, f"multi_atlas_tissue_prior.nii.gz")
    multi_atlas_tissue_seg_path = os.path.join(save_dir, f"multi_atlas_tissue_seg.nii.gz")
    final_parcel_path = os.path.join(save_dir, f"final_parcellation.nii.gz")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    ####################################################################################################################
    # read structure and tissue info
    ####################################################################################################################

    tissue_dict = pd.read_csv(tissue_info_csv_path, index_col="label").to_dict(orient='dict')
    num_tissue = len(tissue_dict['name'])

    structure_df = pd.read_csv(structure_info_csv_path, index_col="label", converters={'tissues': literal_eval})
    structure_dict = structure_df.to_dict(orient='dict')

    ####################################################################################################################
    # check that all labels present in the atlases are present in the structure info csv file
    ####################################################################################################################

    unique_labels = set()
    # number of labels in the atlas segmentations
    for atlas_dict in atlas_paths_dicts_list:
        atlas_seg = nib.load(atlas_dict["seg_path"]).get_fdata(dtype=np.float16).astype(np.uint8)
        unique_labels_curr_atlas = np.unique(atlas_seg)
        unique_labels.update(unique_labels_curr_atlas)

        # check that all labels present in the atlas are present in the structure info csv file
        set_of_labels_not_in_structure_info_csv = set(unique_labels_curr_atlas) - set(structure_df.index)
        assert len(set_of_labels_not_in_structure_info_csv) == 0, f"The following labels are present in the atlas segmentation {atlas_dict['seg_path']}, but not in the structure info csv file {structure_info_csv_path}: \n{set_of_labels_not_in_structure_info_csv}."

        # check if all labels present in the structure info csv file are present in the atlas
        set_of_labels_not_in_atlas = set(structure_df.index) - set(unique_labels_curr_atlas)
        if not len(set_of_labels_not_in_atlas) == 0:
            print(f"The following labels are present in the structure info csv file {structure_info_csv_path}, but not in the atlas segmentation {atlas_dict['seg_path']}: \n{set_of_labels_not_in_atlas}.")

    num_unique_labels_all_atlases = len(unique_labels)
    print(f"Number of unique labels in all atlases: {num_unique_labels_all_atlases}")

    # number of labels in the structure info csv file
    num_label_structure = len(structure_df.index)
    print(f"Number of unique labels in the structure info csv file: {num_label_structure}")

    # length of label dimension
    num_class = num_unique_labels_all_atlases

    # check that the tissues in the structure info csv file are present in the tissue info csv file
    unique_tissues_in_structure_info_csv = np.unique(np.concatenate(structure_df["tissues"].values))
    tissues_in_tissue_info_csv = np.array(list(tissue_dict["name"].keys()))
    set_of_tissues_not_in_tissue_info_csv = set(unique_tissues_in_structure_info_csv) - set(tissues_in_tissue_info_csv)
    assert len(set_of_tissues_not_in_tissue_info_csv) == 0, f"The following tissues are present in the structure info csv file {structure_info_csv_path}, but not in the tissue info csv file {tissue_info_csv_path}: \n{set_of_tissues_not_in_tissue_info_csv}."
    set_of_tissues_not_in_structure_info_csv = set(tissues_in_tissue_info_csv) - set(unique_tissues_in_structure_info_csv)
    if not len(set_of_tissues_not_in_structure_info_csv) == 0:
        print(f"The following tissues are present in the tissue info csv file {tissue_info_csv_path}, but no structure is assigned to this label in the structure info csv file {structure_info_csv_path}: \n{set_of_tissues_not_in_structure_info_csv}.")

    num_unique_tissues_in_structure_info_csv = len(unique_tissues_in_structure_info_csv)
    print(f"Number of unique tissues in the structure info csv file: {num_unique_tissues_in_structure_info_csv}")
    num_unique_tissues_in_tissue_info_csv = len(tissues_in_tissue_info_csv)
    print(f"Number of unique tissues in the tissue info csv file: {num_unique_tissues_in_tissue_info_csv}")

    ####################################################################################################################
    # Register the atlas segmentations to the input image and compute the similarity weights
    ####################################################################################################################

    time_0 = time.time()
    img_affine = nib.load(img_path).affine

    # load the mask
    if mask_path:
        mask_nii = nib.load(mask_path)
        img_mask = mask_nii.get_fdata(dtype=np.float16).astype(np.uint8)
    else:
        img_mask = None

    param_tuples = [(img_path, mask_path, atls["name"], atls["img_path"], atls["seg_path"], atls["mask_path"], save_dir)
                    for atls in atlas_paths_dicts_list]

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
    # calculate the multi-atlas probabilities for each structure  # H x W x D x num_class
    ####################################################################################################################
    print(f"Load the warped atlas segmentations from disk ...")
    t_0_load_warped_atlases = time.time()
    with Pool(NUM_POOLS) as p:
        warped_atlases = np.array(p.starmap(nibabel_load_and_get_fdata, [(warped_atlas_path_list[a], np.uint8) for a in
                                            range(num_atlases)]))  # num_atlases x H x W x D

    print(f"Warped atlas segmentations loading completed after {time.time() - t_0_load_warped_atlases:.3f} seconds")

    multi_atlas_proba_seg = np.zeros((*warped_atlases.shape[1:], num_class), dtype=np.float32)
    multi_atlas_proba_seg = get_multi_atlas_proba_seg(warped_atlases,
                                                      weights,
                                                      np.array(list(unique_labels)),
                                                      multi_atlas_proba_seg)

    # # save the merged probabilities (time consuming, since H x W x D x num_class is large)
    # multi_atlas_proba_seg_nii = nib.Nifti1Image(multi_atlas_proba_seg, affine=img_affine)
    # nib.save(multi_atlas_proba_seg_nii, multi_atlas_proba_seg_path)

    ####################################################################################################################
    # calculate the multi-atlas tissue prior by distributing the probabilities of each structure to its assigned tissues
    ####################################################################################################################
    
    time_0_tissue_prior = time.time()

    multi_atlas_tissue_prior = get_multi_atlas_tissue_prior(multi_atlas_proba_seg,
                                                            np.array(list(unique_labels)),
                                                            structure_dict,
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
                     }

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
    final_parcellation = get_structure_seg_from_tissue_seg(multi_atlas_tissue_seg,
                                                           multi_atlas_proba_seg,
                                                           np.array(list(unique_labels)),
                                                           structure_dict['tissues'])

    print(f"Running structure segmentation completed after {time.time() - t_0_struct_seg:.3f} seconds")

    # save the final parcellation
    predicted_segmentation_nii = nib.Nifti1Image(final_parcellation, img_affine)
    nib.save(predicted_segmentation_nii, final_parcel_path)

    return final_parcellation
