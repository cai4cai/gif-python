import os
import subprocess

import numpy as np
import time

import nibabel as nib
from numba import njit

from src.utils.definitions import NIFTYREG_PATH, NIFTYSEG_PATH

def compute_disp_from_cpp(cpp_path, ref_path, save_disp_path):
    save_folder = os.path.split(save_disp_path)[0]

    # Convert the cpp into a deformation field
    save_def_path = os.path.join(save_folder, 'tmp_def.nii.gz')
    cmd = '%s/reg_transform -ref %s -def %s %s > /dev/null' % (NIFTYREG_PATH, ref_path, cpp_path, save_def_path)
    os.system(cmd)

    # Create the identity transformation to get the displacement
    cpp_id = os.path.join(save_folder, 'output_cpp_identity.nii.gz')
    res_id = os.path.join(save_folder, 'srr_identity.nii.gz')
    cmd = '%s/reg_f3d -ref %s -flo %s -res %s -cpp %s -be 1. -le 0. -ln 3 -voff' % \
          (NIFTYREG_PATH, ref_path, ref_path, res_id, cpp_id)
    os.system(cmd)
    save_id_path = os.path.join(save_folder, 'tmp_id_def.nii.gz')
    cmd = '%s/reg_transform -ref %s -def %s %s > /dev/null' % (NIFTYREG_PATH, ref_path, cpp_id, save_id_path)
    os.system(cmd)

    # Substract the identity to get the displacement field
    deformation_nii = nib.load(save_def_path)
    deformation = deformation_nii.get_fdata().astype(np.float32)
    identity = nib.load(save_id_path).get_fdata().astype(np.float32)
    disp = deformation - identity
    disp_nii = nib.Nifti1Image(disp, deformation_nii.affine)
    nib.save(disp_nii, save_disp_path)


def structure_seg_from_tissue_seg(tiss_seg, lab_probs, tissue_dict):
    """
    Assigns a label to each voxel in tiss_seg based on the highest probability in lab_probs, and the tissue_dict.
    :param tiss_seg: tissue segmentation, shape (x, y, z)
    :param lab_probs: label probabilities, shape (x, y, z, num_labels)
    :param tissue_dict: dictionary mapping labels to tissues, e.g. {0: [0, 1], 1: [2, 3], 2: [4, 5]}
    :return: structure segmentation, shape (x, y, z)
    """

    # time_0 = time.time()

    # lab_probs_idx_sorted = lab_probs.argsort(axis=-1)  # flip to get ascending sort order
    # print('Time for sorting all label probabilities : ', time.time() - time_0)

    # start_time = time.time()
    # num_labels = lab_probs_idx_sorted.shape[-1]
    # num_tissues = np.max(tiss_seg) + 1
    #
    # # reverse tissue_dict by creating a dictionary with tissues as keys and lists of structures as values
    # tissue_dict_rev = {}
    # for l in tissue_dict:
    #     assigned_tissues = tissue_dict[l]
    #     for t in assigned_tissues:
    #         if t in tissue_dict_rev.keys():
    #             tissue_dict_rev[t].append(l)
    #         else:
    #             tissue_dict_rev[t] = [l]
    #
    # # convert tissue seg to one-hot encoding
    # time_0 = time.time()
    # tiss_seg_onehot = np.zeros((*tiss_seg.shape, num_tissues), dtype=bool)
    # for t in range(num_tissues):
    #     tiss_seg_onehot[..., t][tiss_seg == t] = 1
    # print('Time for converting tissue segmentation to one-hot encoding : ', time.time() - time_0)
    #
    #
    # # fill the structure segmentation:
    # # start with the label of highest probability, and check if it maps to the correct tissue according to tissue_dict
    # # and tiss_seg, if so, adopt this label, if not, check the next highest label, and so on until structure_seg is filled
    # # (i.e. no more -1 values)
    # structure_seg = np.ones_like(tiss_seg) * -1
    # i = 0
    # while np.any(structure_seg == -1) and i < num_labels:
    #     print("structure seg fill number:", i, " remaining -1 values:", np.sum(structure_seg == -1))
    #     mask1 = structure_seg == -1  # only consider locations that have not been assigned yet
    #     for t in tissue_dict_rev:
    #         mask2 = np.isin(lab_probs_idx_sorted[..., num_labels - i - 1], tissue_dict_rev[t]) # select locations where the ith-highest label maps to the current tissue
    #         mask3 = tiss_seg_onehot[..., t]  # select locations where the tissue segmentation corresponds to the current tissue
    #
    #         mask = mask1 * mask2 * mask3 # locations for which all conditions are True
    #
    #         lab_idx_curr = lab_probs_idx_sorted[..., num_labels - i - 1]  # map of the ith-highest labels
    #         # copy values for the masked values
    #         structure_seg[mask] = lab_idx_curr[mask]
    #
    #     i += 1


    # alternative method: loop over all voxels and check if the highest label maps to the correct tissue
    # according to tissue_dict and tiss_seg
    num_labels = lab_probs.shape[-1]
    structure_seg = np.zeros_like(tiss_seg)
    range_x = range(tiss_seg.shape[0])
    range_y = range(tiss_seg.shape[1])
    range_z = range(tiss_seg.shape[2])
    for x in range_x:
        for y in range_y:
            for z in range_z:
                    for i in range(num_labels):
                        # get the label with the ith-highest probability
                        #lab_idx_curr = lab_probs_idx_sorted[x, y, z, num_labels - i - 1]
                        # get the label index with the ith-highest probability using np.argpartition
                        lab_idx_curr = np.argpartition(lab_probs[x, y, z, :], -i - 1)[-i - 1]
                        assigned_tissues = tissue_dict[lab_idx_curr]
                        if tiss_seg[x, y, z] in assigned_tissues:
                            structure_seg[x, y, z] = lab_idx_curr
                            break

    return structure_seg


def seg_EM(input_filename, output_filename, mask_filename, prior_filename, verbose_level, max_iterations,
           min_iterations, bias_field_order, bias_field_thresh, mrf_beta):
    """
    Performs EM segmentation on the atlas using niftyseg.
    """

    command = [os.path.join(NIFTYSEG_PATH, 'seg_EM'),
               '-in', input_filename,
               '-out', output_filename,
               '-mask', mask_filename,
               '-priors4D', prior_filename,
               '-v', str(verbose_level),
               '-max_iter', str(max_iterations),
               '-min_iter', str(min_iterations),
               '-bc_order', str(bias_field_order),
               '-bc_thresh', str(bias_field_thresh),
               '-mrf_beta', str(mrf_beta)]

    # Run the command
    subprocess.call(command)