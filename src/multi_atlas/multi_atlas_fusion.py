import numpy as np
from numba import njit, prange


@njit(parallel=True)
def get_multi_atlas_proba_seg(warped_atlases, weights, labels_array, multi_atlas_proba_seg):
    """
    Calculate the multi-atlas probability segmentation by summing the weights of each atlas for each label. The
    probability of each label at a voxel is the sum of the weights of all warped atlases that have that label at that
    voxel.
    :param warped_atlases: num_atlases x H x W x D
    :param weights: num_atlases x H x W x D
    :param labels_array: array of labels to consider (dictionary not used here because numba does not support it)
    :param multi_atlas_proba_seg: output numpy array initialized to zeros of shape H x W x D x num_class (needs to be
        initialized outside of the function because numba does not support numpy.zeros in non-python mode)
    :return: multi_atlas_proba_seg: H x W x D x num_class
    """
    for x in prange(warped_atlases.shape[1]):
        # print(x)
        for y in range(warped_atlases.shape[2]):
            for z in range(warped_atlases.shape[3]):
                for l in labels_array:
                    for a in range(warped_atlases.shape[0]):

                        if warped_atlases[a, x, y, z] == l:
                            multi_atlas_proba_seg[x, y, z, l] += weights[a, x, y, z]
                # where ever the sum of probabilities is 0, set the first class (background) to 1
                if np.sum(multi_atlas_proba_seg[x, y, z, :]) == 0:
                    multi_atlas_proba_seg[x, y, z, 0] = 1

    return multi_atlas_proba_seg


def get_multi_atlas_tissue_prior(
                                multi_atlas_proba_seg,
                                structure_dict,
                                num_class,
                                num_tissue,
                                mask):
    """
    Calculate the multi-atlas tissue prior by distributing the probabilities of each structure to its assigned tissues
    :param multi_atlas_proba_seg: numpy array of probabilities for each structure, shape H x W x D x num_class
    :param structure_dict: dictionary mapping structures to tissues, e.g. {'tissues': {0: [0, 1], 1: [2, 3], 2: [4, 5]}}
    :param num_class: number of structures
    :param num_tissue: number of tissues
    :param mask: target image mask, shape H x W x D

    :return: multi_atlas_tissue_prior: numpy array of prior probabilities for each tissue, shape H x W x D x num_tissue
    """

    multi_atlas_tissue_prior = np.zeros(tuple(multi_atlas_proba_seg.shape[:3] + (num_tissue,)), dtype=np.float32)

    for l in range(num_class):
        if l not in structure_dict['tissues']:
            continue
        assigned_tissues = structure_dict['tissues'][l]
        for t in assigned_tissues:
            multi_atlas_tissue_prior[:, :, :, t] += multi_atlas_proba_seg[:, :, :, l] / len(assigned_tissues) # H x W x D x num_tissue

    # set the background tissue class to the inverse of the mask
    multi_atlas_tissue_prior[:, :, :, 0] = 1-mask

    # set voxels that are not assigned to any tissue to the background tissue
    multi_atlas_tissue_prior[np.sum(multi_atlas_tissue_prior, axis=-1) == 0, 0] = 1

    # normalize the tissue prior
    multi_atlas_tissue_prior = multi_atlas_tissue_prior / np.sum(multi_atlas_tissue_prior, axis=-1, keepdims=True)

    # set everything below 0.01 to 0 and re-normalize
    multi_atlas_tissue_prior[multi_atlas_tissue_prior < 0.01] = 0

    # set voxels that are not assigned to any tissue to the background tissue
    multi_atlas_tissue_prior[np.sum(multi_atlas_tissue_prior, axis=-1) == 0, 0] = 1

    # re-normalize
    multi_atlas_tissue_prior = multi_atlas_tissue_prior / np.sum(multi_atlas_tissue_prior, axis=-1, keepdims=True)

    return multi_atlas_tissue_prior


def get_structure_seg_from_tissue_seg(tiss_seg, lab_probs, assigned_tissues_dict):
    """
    Assigns a label to each voxel in tiss_seg based on the highest probability in lab_probs, and the assigned_tissues_dict.
    That means for each voxel, the highest label from lab_probs is checked against the assigned_tissues_dict to see if
    any of the assigned tissues match the tissue at that voxel in tiss_seg. If there is a match, the label is assigned
    to that voxel in structure_seg.

    :param tiss_seg: tissue segmentation, shape (x, y, z)
    :param lab_probs: label probabilities, shape (x, y, z, num_labels)
    :param assigned_tissues_dict: dictionary mapping labels to tissues, e.g. {0: [0, 1], 1: [2, 3], 2: [4, 5]}

    :return: structure segmentation, shape (x, y, z)
    """
    # loop over all voxels and check if the highest label maps to the correct tissue
    # according to assigned_tissues_dict and tiss_seg
    num_labels = lab_probs.shape[-1]
    structure_seg = np.zeros_like(tiss_seg)
    range_x = range(tiss_seg.shape[0])
    range_y = range(tiss_seg.shape[1])
    range_z = range(tiss_seg.shape[2])
    for x in range_x:
        for y in range_y:
            for z in range_z:
                    for i in range(num_labels):
                        # get the label index with the ith-highest probability using np.argpartition
                        lab_idx_curr = np.argpartition(lab_probs[x, y, z, :], -i - 1)[-i - 1]
                        assigned_tissues = assigned_tissues_dict[lab_idx_curr]
                        if tiss_seg[x, y, z] in assigned_tissues:
                            structure_seg[x, y, z] = lab_idx_curr
                            break

    return structure_seg
