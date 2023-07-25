import numpy as np
from numba import njit, prange, jit
from scipy.ndimage import gaussian_filter
import nibabel as nib
import os


@njit(parallel=True)
def get_multi_atlas_proba_seg(warped_atlases, weights, labels_array, multi_atlas_proba_seg):
    for x in prange(warped_atlases.shape[1]):
        print(x)
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


def get_multi_atlas_tissue_prior(multi_atlas_proba_seg,
                                    structure_dict,
                                    num_class,
                                    num_tissue,
                                    mask_nii):
    """
    Calculate the multi-atlas tissue prior by distributing the probabilities of each structure to its assigned tissues
    :param multi_atlas_proba_seg: H x W x D x num_class
    :param structure_dict: dict
    :param num_tissue: int
    :param img_nii: nibabel image
    :param mask_nii: nibabel image
    :return: multi_atlas_tissue_prior: H x W x D x num_tissue
    """

    multi_atlas_tissue_prior = np.zeros(tuple(multi_atlas_proba_seg.shape[:3] + (num_tissue,)), dtype=np.float32)

    for l in range(num_class):
        if not l in structure_dict['tissues']:
            continue
        assigned_tissues = structure_dict['tissues'][l]
        for t in assigned_tissues:
            multi_atlas_tissue_prior[:, :, :, t] += multi_atlas_proba_seg[:, :, :, l] / len(assigned_tissues) # H x W x D x num_tissue


    # set the background tissue class to the inverse of the mask
    multi_atlas_tissue_prior[:, :, :, 0] = 1-mask_nii.get_fdata(dtype=np.float16).astype(np.uint8)

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

    return multi_atlas_tissue_prior

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
        V[mask==0] = 0
        VV = gaussian_filter(V, **kwargs)

        # second convolution (with ones inside the mask and zeros outside the mask)
        W = np.ones_like(input)
        W[mask==0] = 0
        WW = gaussian_filter(W, **kwargs)

        # divide the two convolutions and mask the result
        smoothed = (VV / WW) * mask

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

    # save the smoothed images
    nib.save(nib.Nifti1Image(image_mean, affine=affine), os.path.join(save_folder_path, 'image_mean.nii.gz'))
    nib.save(nib.Nifti1Image(image_squ, affine=affine), os.path.join(save_folder_path, 'image_squ.nii.gz'))
    nib.save(nib.Nifti1Image(atlas_mean, affine=affine), os.path.join(save_folder_path, 'atlas_mean.nii.gz'))
    nib.save(nib.Nifti1Image(atlas_squ, affine=affine), os.path.join(save_folder_path, 'atlas_squ.nii.gz'))
    nib.save(nib.Nifti1Image(image_atlas_prod, affine=affine), os.path.join(save_folder_path, 'image_atlas_prod.nii.gz'))

    # compute the standard deviations
    image_std = image_squ - image_mean * image_mean
    atlas_std = atlas_squ - atlas_mean * atlas_mean

    covar = image_atlas_prod - image_mean * atlas_mean
    variances_prod = np.sqrt(image_std * atlas_std)

    lncc = covar / variances_prod
    lncc_distance = 1.0 - lncc

    # save lncc_distance
    nib.save(nib.Nifti1Image(lncc_distance, affine=affine), os.path.join(save_folder_path, 'lncc_distance.nii.gz'))

    return lncc_distance