"""GIF-like multi-atlas segmentation of a target image."""

from __future__ import annotations

import logging
import time
from ast import literal_eval
from pathlib import Path
from typing import Dict, List, Optional, Union

import nibabel as nib
import numpy as np
import pandas as pd

from src.multi_atlas.atlas_registration_and_resampling import warp_atlas_and_calc_similarity_weights
from src.multi_atlas.multi_atlas_fusion import (
    get_multi_atlas_proba_seg,
    get_multi_atlas_tissue_prior,
    get_structure_seg_from_tissue_seg,
)
from src.multi_atlas.utils import nibabel_load_and_get_fdata, seg_EM
from src.utils.config import GifConfig, default_config
from src.utils.nifti import relax_quaternion_check

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _require_file(path: PathLike, what: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{what} does not exist: {path}")


def multi_atlas_segmentation(
    img_path: PathLike,
    mask_path: Optional[PathLike],
    atlas_paths_dicts_list: List[Dict[str, str]],
    structure_info_csv_path: PathLike,
    tissue_info_csv_path: PathLike,
    save_dir: PathLike,
    config: Optional[GifConfig] = None,
) -> np.ndarray:
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
    :param config: GifConfig with the parameters of the pipeline (default: ``default_config()``)
    :return: the final parcellation (also saved as ``final_parcellation.nii.gz`` in ``save_dir``)
    :raises FileNotFoundError: if an input file does not exist
    :raises ValueError: if the structure/tissue information is inconsistent with the atlases
    :raises ExternalToolError: if a NiftyReg or NiftySeg tool fails
    """
    from multiprocessing import Pool  # imported here: nothing is imported or started at module import

    config = config or default_config()
    relax_quaternion_check()

    logger.info("Multi-atlas segmentation using the GIF-like fusion method...")
    ####################################################################################################################
    # Check inputs
    ####################################################################################################################
    logger.info("Checking inputs...")
    time_0 = time.time()
    _require_file(img_path, "Input image")
    if mask_path is not None:
        _require_file(mask_path, "Input mask")
    else:
        logger.info("No input mask for the target image was provided...")

    # work on copies: the caller's dictionaries are not modified
    atlas_paths_dicts_list = [dict(atlas_dict) for atlas_dict in atlas_paths_dicts_list]
    for atlas_dict in atlas_paths_dicts_list:
        _require_file(atlas_dict["img_path"], "Atlas image")
        _require_file(atlas_dict["seg_path"], "Atlas segmentation")
        if "mask_path" in atlas_dict:
            _require_file(atlas_dict["mask_path"], "Atlas mask")
        else:
            atlas_dict["mask_path"] = None
            logger.info("No input mask for atlas %s was provided...", atlas_dict["name"])

    _require_file(structure_info_csv_path, "Structure info csv file")
    _require_file(tissue_info_csv_path, "Tissue info csv file")

    logger.info("Inputs checked in %.2f seconds.", time.time() - time_0)
    ####################################################################################################################
    # Prepare output paths
    ####################################################################################################################
    save_dir = Path(save_dir)
    multi_atlas_proba_seg_argmax_path = save_dir / "multi_atlas_proba_seg_argmax.nii.gz"
    multi_atlas_tissue_prior_path = save_dir / "multi_atlas_tissue_prior.nii.gz"
    multi_atlas_tissue_seg_path = save_dir / "multi_atlas_tissue_seg.nii.gz"
    final_parcel_path = save_dir / "final_parcellation.nii.gz"

    save_dir.mkdir(parents=True, exist_ok=True)

    ####################################################################################################################
    # read structure and tissue info
    ####################################################################################################################

    tissue_dict = pd.read_csv(tissue_info_csv_path, index_col="label").to_dict(orient="dict")
    num_tissue = len(tissue_dict["name"])

    structure_df = pd.read_csv(structure_info_csv_path, index_col="label", converters={"tissues": literal_eval})
    structure_dict = structure_df.to_dict(orient="dict")

    ####################################################################################################################
    # check that all labels present in the atlases are present in the structure info csv file
    ####################################################################################################################
    logger.info("Checking that all labels present in the atlases are present in the structure info csv file...")
    time_0 = time.time()
    unique_labels = set()
    # number of labels in the atlas segmentations
    for atlas_dict in atlas_paths_dicts_list:
        atlas_seg = nib.load(str(atlas_dict["seg_path"])).get_fdata(dtype=np.float16).astype(np.uint16)
        unique_labels_curr_atlas = np.unique(atlas_seg)
        unique_labels.update(unique_labels_curr_atlas)

        # check that all labels present in the atlas are present in the structure info csv file
        set_of_labels_not_in_structure_info_csv = set(unique_labels_curr_atlas) - set(structure_df.index)
        if set_of_labels_not_in_structure_info_csv:
            raise ValueError(
                f"The following labels are present in the atlas segmentation {atlas_dict['seg_path']}, "
                f"but not in the structure info csv file {structure_info_csv_path}: "
                f"{sorted(int(label) for label in set_of_labels_not_in_structure_info_csv)}."
            )

        # check if all labels present in the structure info csv file are present in the atlas
        set_of_labels_not_in_atlas = set(structure_df.index) - set(unique_labels_curr_atlas)
        if set_of_labels_not_in_atlas:
            logger.info(
                "The following labels are present in the structure info csv file %s, but not in the atlas "
                "segmentation %s: %s.",
                structure_info_csv_path,
                atlas_dict["seg_path"],
                set_of_labels_not_in_atlas,
            )

    num_unique_labels_all_atlases = len(unique_labels)
    logger.info("Number of unique labels in all atlases: %d", num_unique_labels_all_atlases)

    # label order (the position in this array is the class index of the probability maps)
    labels_array = np.array(list(unique_labels))

    # mapping from index to label
    index_to_label = dict(enumerate(unique_labels))

    logger.info("Number of unique labels in the structure info csv file: %d", len(structure_df.index))

    # length of label dimension
    num_class = num_unique_labels_all_atlases

    # check that the tissues in the structure info csv file are present in the tissue info csv file
    unique_tissues_in_structure_info_csv = np.unique(np.concatenate(structure_df["tissues"].values))
    tissues_in_tissue_info_csv = np.array(list(tissue_dict["name"].keys()))
    set_of_tissues_not_in_tissue_info_csv = set(unique_tissues_in_structure_info_csv) - set(tissues_in_tissue_info_csv)
    if set_of_tissues_not_in_tissue_info_csv:
        raise ValueError(
            f"The following tissues are present in the structure info csv file {structure_info_csv_path}, "
            f"but not in the tissue info csv file {tissue_info_csv_path}: "
            f"{sorted(int(t) for t in set_of_tissues_not_in_tissue_info_csv)}."
        )
    set_of_tissues_not_in_structure_info_csv = set(tissues_in_tissue_info_csv) - set(
        unique_tissues_in_structure_info_csv
    )
    if set_of_tissues_not_in_structure_info_csv:
        logger.info(
            "The following tissues are present in the tissue info csv file %s, but no structure is assigned to "
            "this label in the structure info csv file %s: %s.",
            tissue_info_csv_path,
            structure_info_csv_path,
            set_of_tissues_not_in_structure_info_csv,
        )

    logger.info(
        "Number of unique tissues in the structure info csv file: %d", len(unique_tissues_in_structure_info_csv)
    )
    logger.info("Number of unique tissues in the tissue info csv file: %d", len(tissues_in_tissue_info_csv))
    logger.info("Finished checks. Time elapsed: %.2f seconds.", time.time() - time_0)

    ####################################################################################################################
    # Register the atlas segmentations to the input image and compute the similarity weights
    ####################################################################################################################

    time_0 = time.time()
    img_affine = nib.load(str(img_path)).affine

    # load the mask
    if mask_path:
        img_mask = nib.load(str(mask_path)).get_fdata(dtype=np.float16).astype(np.uint16)
    else:
        img_mask = None

    param_tuples = [
        (img_path, mask_path, atls["name"], atls["img_path"], atls["seg_path"], atls["mask_path"], save_dir, config)
        for atls in atlas_paths_dicts_list
    ]

    if config.multiprocessing:
        with Pool(config.num_pools) as p:
            logger.info("Start multiprocessing with %d pools...", config.num_pools)
            out = p.starmap(warp_atlas_and_calc_similarity_weights, param_tuples)
    else:
        out = [warp_atlas_and_calc_similarity_weights(*params) for params in param_tuples]

    warped_atlas_path_list = [p[0] for p in out]
    weights_path_list = [p[1] for p in out]

    logger.info("Registration and similarity weights calculation completed after %.3f seconds", time.time() - time_0)

    ####################################################################################################################
    # Merging probabilities
    ####################################################################################################################
    # Weight and merge the atlas segmentations to get the multi-atlas probabilities for each structure
    logger.info("Merging probabilities...")
    num_atlases = len(warped_atlas_path_list)

    # Load the weights and the atlas segmentations
    logger.info("Load weights from disk...")
    t_0_weight = time.time()

    if config.multiprocessing:
        with Pool(config.num_pools) as p:
            weights_list = p.map(nibabel_load_and_get_fdata, weights_path_list)  # n_atlas * [H, W, D]
    else:
        weights_list = [nibabel_load_and_get_fdata(path) for path in weights_path_list]

    # remove the weight files
    for path in weights_path_list:
        Path(path).unlink()

    # calculate the weights from the heat kernels
    weights = np.stack(weights_list, axis=0)  # n_atlas, H, W, D
    # set NaN in the weights to 0
    weights[np.isnan(weights)] = 0

    logger.info("Weights loading completed after %.3f seconds", time.time() - t_0_weight)

    ####################################################################################################################
    # calculate the multi-atlas probabilities for each structure  # H x W x D x num_class
    ####################################################################################################################
    logger.info("Load the warped atlas segmentations from disk ...")
    t_0_load_warped_atlases = time.time()
    warped_atlas_load_args = [(warped_atlas_path_list[a], np.uint16) for a in range(num_atlases)]
    if config.multiprocessing:
        with Pool(config.num_pools) as p:
            warped_atlases = np.array(p.starmap(nibabel_load_and_get_fdata, warped_atlas_load_args))
    else:
        warped_atlases = np.array([nibabel_load_and_get_fdata(*args) for args in warped_atlas_load_args])
    # (num_atlases x H x W x D)

    logger.info(
        "Warped atlas segmentations loading completed after %.3f seconds", time.time() - t_0_load_warped_atlases
    )

    # calculate the multi-atlas probabilities for each structure
    logger.info("Calculate the multi-atlas probabilities for each structure...")
    time_0 = time.time()
    multi_atlas_proba_seg = np.zeros((*warped_atlases.shape[1:], num_class), dtype=np.float32)
    multi_atlas_proba_seg = get_multi_atlas_proba_seg(warped_atlases, weights, labels_array, multi_atlas_proba_seg)

    # get the maximum index of the probabilities
    multi_atlas_seg = np.argmax(multi_atlas_proba_seg, axis=-1).astype(np.uint16)

    # map the maximum index to the corresponding original labels using index_to_label dict
    multi_atlas_seg = np.vectorize(index_to_label.get)(multi_atlas_seg)

    nib.save(nib.Nifti1Image(multi_atlas_seg, affine=img_affine), str(multi_atlas_proba_seg_argmax_path))

    logger.info("Multi-atlas structure probabilities merging completed after %.3f seconds", time.time() - time_0)
    ####################################################################################################################
    # calculate the multi-atlas tissue prior by distributing the probabilities of each structure to its assigned tissues
    ####################################################################################################################

    time_0_tissue_prior = time.time()

    multi_atlas_tissue_prior = get_multi_atlas_tissue_prior(
        multi_atlas_proba_seg,
        labels_array,
        structure_dict,
        num_tissue,
        img_mask,
    )

    # save tissue prior
    logger.info("Saving multi-atlas tissue prior...")
    nib.save(nib.Nifti1Image(multi_atlas_tissue_prior, affine=img_affine), str(multi_atlas_tissue_prior_path))

    logger.info("Multi-atlas tissue prior calculation completed after %.3f seconds", time.time() - time_0_tissue_prior)

    ####################################################################################################################
    # run seg_EM algorithm to get final tissue segmentation
    ####################################################################################################################

    logger.info("Running seg_EM algorithm...")
    t_0_segEM = time.time()
    seg_EM(
        input_filename=img_path,
        output_filename=multi_atlas_tissue_seg_path,
        mask_filename=mask_path,
        prior_filename=multi_atlas_tissue_prior_path,
        config=config,
    )

    # load the result
    logger.info("Loading seg_EM result...")
    multi_atlas_tissue_seg_EM = nib.load(str(multi_atlas_tissue_seg_path)).get_fdata(dtype=np.float16)

    # where the prior background is 1, set the multi_atlas_tissue_seg to 1
    # this is to avoid artifacts at the image boundary, where the seg_EM algorithm reduces the background probability
    multi_atlas_tissue_seg_EM[multi_atlas_tissue_prior[..., 0] == 1] = 1

    # at the image boundary, the seg_EM algorithm reduces the background probability, which leads to artifacts
    # set the background probability to 1 at the image boundary
    multi_atlas_tissue_seg_EM[[0, -1], :, :] = 1
    multi_atlas_tissue_seg_EM[:, [0, -1], :] = 1
    multi_atlas_tissue_seg_EM[:, :, [0, -1]] = 1

    # argmax to get the final tissue segmentation
    multi_atlas_tissue_seg = np.argmax(multi_atlas_tissue_seg_EM, axis=-1).astype(np.uint16)

    logger.info("Running seg_EM algorithm completed after %.3f seconds", time.time() - t_0_segEM)

    ####################################################################################################################
    # get the final structure segmentation
    ####################################################################################################################

    # get the label with the maximum probability according to multi_atlas_proba_seg under the condition that one of
    # the assigned tissues corresponds to the tissue in multi_atlas_tissue_seg
    logger.info("Running structure segmentation...")
    t_0_struct_seg = time.time()
    final_parcellation = get_structure_seg_from_tissue_seg(
        multi_atlas_tissue_seg,
        multi_atlas_proba_seg,
        labels_array,
        structure_dict["tissues"],
    )

    logger.info("Running structure segmentation completed after %.3f seconds", time.time() - t_0_struct_seg)

    # save the final parcellation
    nib.save(nib.Nifti1Image(final_parcellation, img_affine), str(final_parcel_path))

    return final_parcellation
