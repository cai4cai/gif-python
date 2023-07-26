import os
import subprocess

import numpy as np
import nibabel as nib

from scipy.ndimage import gaussian_filter

from src.utils.definitions import NIFTYREG_PATH, NIFTYSEG_PATH


def nibabel_load_and_get_fdata(filepath, dtype=np.float32):
    if dtype == np.uint8:
        return nib.load(filepath).get_fdata(dtype=np.float16).astype(dtype)
    else:
        return nib.load(filepath).get_fdata(dtype=dtype).astype(dtype)


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


def gaussian_smooth_with_mask(input, mask, **kwargs):
        """
        Smooth an image with a Gaussian kernel, ignoring values outside the mask.
        In principle the kernel weight will be replaced by zero wherever it overlaps with a zero in the mask.
        Rather than changing the kernel an equivalent approach can be used by performing two convolutions as explained
        here: https://stackoverflow.com/a/36307291

        :param input: input image
        :param mask: input image mask
        :param kwargs: arguments for gaussian_filter
        :return: smoothed image
        """

        # first convolution (with image intensities in the mask and zeros outside the mask)
        V = input.copy()
        V[mask == 0] = 0
        VV = gaussian_filter(V, **kwargs)

        # second convolution (with ones inside the mask and zeros outside the mask)
        W = np.ones_like(input)
        W[mask == 0] = 0
        WW = gaussian_filter(W, **kwargs)

        # avoid division by zero
        WW[np.invert(mask.astype(bool))] = 1

        # divide the two convolutions and mask the result
        smoothed = (VV * mask / WW)

        return smoothed


def get_lncc_distance(image, mask, atlas_warped_image, save_folder_path, affine, spacing):
    '''
    Compute the Local Normalized Cross Correlation (LNCC) distance between the input image and the atlas image.
    LNCC is defined as local covariance divided by the square root of the product of local variances.
    covariance(I, A) = mean(I*A) - mean(I)*mean(A)
    variance(I) = mean(I^2) - mean(I)^2
    The weighted means are computed in a local neighborhood.

    The LNCC distance is defined as 1 - LNCC, where LNCC is the Local Normalized Cross Correlation.
    The LNCC is computed using a Gaussian kernel with standard deviation specified by kernel_std.
    Relationship between LNCC and Gaussian smoothing is described for example in the following paper:
    https://discovery.ucl.ac.uk/id/eprint/1501070/1/paper888.pdf

    :param image: input image
    :param mask: input image mask
    :param atlas_warped_image: atlas image warped to the input image space
    :param save_folder_path: path to the folder where the LNCC distance will be saved
    :param affine: affine matrix of the input image
    :param spacing: spacing of the input image
    :return: LNCC distance
    '''

    # Gaussian kernel standard deviation in mm (if > 0) or in voxels (if < 0)
    kernel_std = [-2.5, -2.5, -2.5]
    # convert kernel standard deviation from mm to voxels
    kernel_std = np.array([abs(k/s) if k < 0 else k for k, s in zip(kernel_std, spacing)])

    # kernel radius in voxels
    kernel_radius = np.floor([s*3 for s in kernel_std]).astype(int)

    # define quantities to be averaged/smoothed for the LNCC computation
    image_mean = image
    image_squ = image * image
    atlas_mean = atlas_warped_image
    atlas_squ = atlas_warped_image * atlas_warped_image
    image_atlas_prod = image * atlas_warped_image

    # create a combined mask of the input image mask and a new mask obtained by excluding NaNs from the atlas image
    fg_mask_combined = np.logical_and(mask, ~np.isnan(atlas_warped_image)).astype(np.float32)

    # smooth the images with a Gaussian kernel
    smoothing_params = {'sigma': kernel_std,
                        'radius': kernel_radius}

    image_mean = gaussian_smooth_with_mask(image_mean, fg_mask_combined, **smoothing_params)
    image_squ = gaussian_smooth_with_mask(image_squ, fg_mask_combined, **smoothing_params)
    atlas_mean = gaussian_smooth_with_mask(atlas_mean, fg_mask_combined, **smoothing_params)
    atlas_squ = gaussian_smooth_with_mask(atlas_squ, fg_mask_combined, **smoothing_params)
    image_atlas_prod = gaussian_smooth_with_mask(image_atlas_prod, fg_mask_combined, **smoothing_params)

    # compute the standard deviations
    image_std = image_squ - image_mean * image_mean
    atlas_std = atlas_squ - atlas_mean * atlas_mean

    covar = image_atlas_prod - image_mean * atlas_mean
    variances_prod = np.sqrt(image_std * atlas_std)

    # avoid division by zero (NaNs will be set to background class later on)
    variances_prod[variances_prod == 0] = np.nan

    lncc = covar / variances_prod
    lncc_distance = 1.0 - lncc

    # # save lncc_distance
    # nib.save(nib.Nifti1Image(lncc_distance, affine=affine), os.path.join(save_folder_path, 'lncc_distance.nii.gz'))

    return lncc_distance


def seg_EM(input_filename,
           output_filename,
           mask_filename,
           prior_filename,
           verbose_level,
           max_iterations,
           min_iterations,
           bias_field_order,
           bias_field_thresh,
           mrf_beta):
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