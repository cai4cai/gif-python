import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import nibabel as nib
import os
from astropy.convolution import convolve

SIGMA_HIGH_PASS_FILTER = 20  # in mm. In the GIF paper they use 20mm.
# NORMALIZATION = 'percentiles'  # can also be 'z_score'
NORMALIZATION = 'z_score'  # z_score with mask erosion works best

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


def log_heat_kernel_GIF(image, mask, atlas_warped_image, atlas_warped_mask, deformation_field, spacing=[0.8]*3):
    def normalize_image(img, brain_mask, mode='z_score'):
        # Prepare brain mask
        fg_mask = (brain_mask > 0)
        fg_mask[np.isnan(img)] = False  # remove NaNs from the mask
        # Erode the mask because we want to make sure we do not include
        # the background voxels to compute the intensity statistic
        fg_mask_eroded = binary_erosion(fg_mask, iterations=3)

        # Compute useful image intensity stats
        img_fg = img[fg_mask_eroded]
        p999 = np.percentile(img_fg, 99.9)
        img_fg[img_fg > p999] = p999
        mean = np.mean(img_fg)

        # Normalize the image
        img[np.isnan(img)] = mean  # Set NaNs to mean values to allow > comparison
        img[img > p999] = p999
        if mode == 'z_score':
            std = np.std(img_fg)
            img = (img - mean) / std
        else:
            median = np.median(img_fg)
            p95 = np.percentile(img_fg, 95)
            p5 = np.percentile(img_fg, 5)
            print('Use percentiles normalization')
            img = (img - median) / (p95 - p5)

        # Set background voxels to 0
        img[np.logical_not(fg_mask)] = 0

        return img

    def apply_cubic_Bsplines_kernel(intensity_map):
        kernel1d = np.array([1./6, 2./3, 1./6])
        kernel3d = kernel1d[:,None,None] * kernel1d[None,:,None] * kernel1d[None,None,:]  # 3x3x3
        output = convolve(intensity_map, kernel3d, mode='nearest')
        return output

    def high_pass_filter(vector_map):
        vector_map = np.squeeze(vector_map)
        sigma = np.array([SIGMA_HIGH_PASS_FILTER / spacing[i] for i in range(3)] + [0.])
        low_fq = gaussian_filter(vector_map, sigma=sigma, mode='nearest')
        output = vector_map - low_fq
        return output

    # Normalize the input image
    img_norm = normalize_image(
        image, mask, mode=NORMALIZATION)

    # Normalize the atlas image intensity (zero mean, unit variance for each volume)
    atlas_img_norm = normalize_image(
        atlas_warped_image, atlas_warped_mask,mode=NORMALIZATION)

    # Compute the intensity term (LSSD)
    ssd = (atlas_img_norm - img_norm) ** 2
    lssd = apply_cubic_Bsplines_kernel(ssd)

    # Remove the low frequencies of the deformations
    disp = high_pass_filter(deformation_field)

    # Compute the displacement field norm (in mm)
    disp_norm = np.linalg.norm(disp, axis=-1)

    # Compute the heat kernel maps
    distance_map = 0.5 * lssd + 0.5 * disp_norm
    log_heat_kernel = -distance_map**2

    return log_heat_kernel, lssd, disp_norm


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