import os
import nibabel as nib
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_dilation
import numpy as np
from time import time
from src.utils.definitions import NIFTYREG_PATH, RESAMPLE_METHOD

SIGMA = 0  # sigma for smoothing the segmentation prior
OMP = 8


def probabilistic_segmentation_prior(image_nii, mask_nii,
                                     template_nii, template_seg_nii, template_mask_nii,
                                     atlas_seg_onehot_path, warped_altas_seg_onehot_path, warped_atlas_img_path,
                                     num_class,
                                     mask_dilation=3, save_folder_path=None, use_affine=True,
                                     affine_only=False, grid_spacing=4, be=0.001, le=0.01, lp=3):
    """
    Summary of what this function does:
    1. register the template to the image (use concatenation with one-hot segmentation to help)
    2. propagate the labels and (optionally) smooth the prior

    The segmentation is used during registration with MSE if seg_nii is not None.
    """
    time_0 = time()

    # Create the folder where to save the registration output
    tmp_folder = './tmp'
    if save_folder_path is None:  # in this case the tmp folder will be deleted
        save_folder_path = tmp_folder

    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    # # Mask the image
    # masked_image_nii = _mask_image(
    #     image_nii=image_nii,
    #     mask_nii=mask_nii,
    #     num_dilation=mask_dilation
    # )

    # Register the template to the image
    affine_params_path, cpp_params_path = _register_atlas_to_img(
        image_nii=image_nii,
        mask_nii=mask_nii,
        atlas_nii=template_nii,
        atlas_mask_nii=template_mask_nii,
        warped_atlas_img_path=warped_atlas_img_path,
        grid_spacing=grid_spacing,
        be=be,
        le=le,
        lp=lp,
        save_folder=save_folder_path,
        use_affine=use_affine,
        affine_only=affine_only,
    )

    # Propagate the labels
    warped_atlas_path_or_prob_seg_l_paths, warped_atlas_seg_mask = _propagate_labels(
        num_class=num_class,
        atlas_seg_nii=template_seg_nii,
        image_nii=image_nii,
        aff_path=affine_params_path,  # will be None if use_affine == False
        cpp_path=cpp_params_path,
        save_folder=save_folder_path,
        atlas_seg_onehot_path=atlas_seg_onehot_path,
        warped_altas_seg_onehot_path=warped_altas_seg_onehot_path,
    )

    # Delete the tmp folder
    if os.path.exists(tmp_folder):
        os.system(f'rm -r "{tmp_folder}"')
    duration = int(time() - time_0)  # in seconds
    minutes = duration // 60
    seconds = duration - minutes * 60
    print('The atlas propagation has been performed in %dmin%dsec' % (minutes, seconds))

    return warped_atlas_path_or_prob_seg_l_paths, warped_atlas_seg_mask


def _mask_image(image_nii, mask_nii, num_dilation):
    image_np = image_nii.get_fdata().astype(np.float32)
    mask_np = mask_nii.get_fdata().astype(np.uint8)
    if num_dilation > 0:
        dilated_mask_np = binary_dilation(mask_np, iterations=num_dilation).astype(np.uint8)
    else:
        dilated_mask_np = mask_np
    image_np[dilated_mask_np == 0] = 0
    out_img_nii = nib.Nifti1Image(image_np, image_nii.affine, image_nii.header)
    return out_img_nii

def _register_atlas_to_img(image_nii, mask_nii,
                           atlas_nii, atlas_mask_nii,
                           warped_atlas_img_path,
                           grid_spacing, be, le, lp,
                           save_folder, use_affine, affine_only):
    """
    Affine registration + non-linear registration with stationary velocity fields
    are performed to register the atlas to the image.
    The segmentation is used with MSE if segmentation_nii is not None.
    """

    # TODO: remove this part or make consistent
    if os.path.isfile(os.path.join(save_folder, 'outputAffine.txt')) \
            and os.path.isfile(os.path.join(save_folder, 'cpp.nii.gz')) \
            and os.path.isfile(warped_atlas_img_path):
        print("Found registration files... Skip registration...")
        return os.path.join(save_folder, 'outputAffine.txt'), os.path.join(save_folder, 'cpp.nii.gz')

    # Return the path to the output velocity field returned by NiftyReg
    def save_nifti(volume_np, affine, save_path):
        volume_nii = nib.Nifti1Image(volume_np, affine)
        nib.save(volume_nii, save_path)

    print(f"Use {OMP} threads in each pool")

    # Prepare the volumes to register
    img_path = os.path.join(save_folder, 'img.nii.gz')
    mask_path = os.path.join(save_folder, 'mask.nii.gz')
    atlas_img_path = os.path.join(save_folder, 'atlas_img.nii.gz')
    atlas_mask_path = os.path.join(save_folder, 'atlas_mask.nii.gz')
    save_nifti(image_nii.get_fdata(), image_nii.affine, img_path)
    save_nifti(atlas_nii.get_fdata(), atlas_nii.affine, atlas_img_path)
    save_nifti(
        mask_nii.get_fdata().astype(np.uint8),
        mask_nii.affine,
        mask_path
    )
    save_nifti(
        atlas_mask_nii.get_fdata().astype(np.uint8),
        atlas_mask_nii.affine,
        atlas_mask_path
    )

    # Affine registration
    if use_affine or affine_only:
        affine_path = os.path.join(save_folder, 'outputAffine.txt')
        affine_res_path = os.path.join(save_folder, 'affine_warped_atlas_img.nii.gz')
        affine_reg_cmd = (
            f'{NIFTYREG_PATH}/reg_aladin '
            f'-ref "{img_path}" '
            f'-rmask "{mask_path}" '
            f'-flo "{atlas_img_path}" '
            f'-fmask "{atlas_mask_path}" '
            f'-res "{affine_res_path}" '
            f'-aff "{affine_path}" '
            f'-comm '
            f'-voff '
            f'-omp {OMP} '
            f'-lp 2 '
            f'-speeeeed '
        )
        print(affine_reg_cmd)
        os.system(affine_reg_cmd)

        # # Warp the atlas image (because the output of reg_aladin is masked)
        # affine_warp_cmd = (
        #     f'{NIFTYREG_PATH}/reg_resample '
        #     f'-ref "{img_path}" '
        #     f'-flo "{atlas_img_path}" '
        #     f'-trans "{affine_path}" '
        #     f'-res "{affine_res_path}" '
        #     f'-inter 1 '
        #     f'-voff '
        #     f'-omp {OMP}'
        # )
        # os.system(affine_warp_cmd)
        #
        # # Warp the atlas mask
        # affine_res_mask_path = os.path.join(save_folder, 'affine_warped_atlas_mask.nii.gz')
        # affine_warp_mask_cmd = (
        #     f'{NIFTYREG_PATH}/reg_resample '
        #     f'-ref "{img_path}" '
        #     f'-flo "{atlas_mask_path}" '
        #     f'-trans "{affine_path}" '
        #     f'-res "{affine_res_mask_path}" '
        #     f'-inter 0 '
        #     f'-voff '
        #     f'-omp {OMP}'
        # )
        # os.system(affine_warp_mask_cmd)

        if affine_only:
            return affine_path, None

    else:  # no affine transformation
        affine_path = None
        affine_res_path = atlas_img_path
        affine_res_mask_path = atlas_mask_path

    # Registration
    res_path = warped_atlas_img_path
    cpp_path = os.path.join(save_folder, 'cpp.nii.gz')
    reg_options = (
        f'-jl 0.0001 '  # Weight of log of the Jacobian determinant penalty term
        f'-be 0.005 '  # Weight of the bending energy (second derivative of the transformation) penalty term  
        f'-maxit 250 '  # Maximum number of iterations
        f'-ln 4 '  # Number of level to perform
        f'-lp 3 '  # Only perform the first levels [ln]
        f'-sx -5.0 '  # Final grid spacing in the x direction, adopted in y and z directions if not specified
        f'-lncc 0 5.0 '
    )
    reg_cmd = (
        f'{NIFTYREG_PATH}/reg_f3d '
        f'-ref "{img_path}" '
        f'-flo "{atlas_img_path}" '
        f'-rmask "{mask_path}" '
        f'-fmask "{atlas_mask_path}" '
        f'-aff "{affine_path}" '
        f'{reg_options} '
        f'-omp {OMP} '
        f'-res "{res_path}" '  # Filename of the resampled image
        f'-cpp "{cpp_path}" '  # Filename of control point grid [outputCPP.nii]
        f'-voff'
    )

    # print('Non linear registration command line:')
    # print(reg_cmd)
    os.system(reg_cmd)
    return affine_path, cpp_path


def _propagate_labels(num_class, atlas_seg_nii, image_nii, aff_path, cpp_path, save_folder, atlas_seg_onehot_path, warped_altas_seg_onehot_path):
    # Infere the tmp folder from input
    if cpp_path is not None:
        tmp_folder = os.path.split(cpp_path)[0]
    else:
        tmp_folder = os.path.split(aff_path)[0]
    image_path = os.path.join(tmp_folder, 'img.nii.gz')
    nib.save(image_nii, image_path)

    # Smooth labels and save them separately
    atlas_seg = atlas_seg_nii.get_fdata().astype(np.uint8)


    if RESAMPLE_METHOD == 0:
        if SIGMA > 0:
            raise Exception("Must not smooth multi-label volume...")

        # Save the atlas seg as input for reg_resample
        atlas_seg_path = os.path.join(save_folder, "atlas_seg.nii.gz")
        atlas_seg_nii = nib.Nifti1Image(atlas_seg, atlas_seg_nii.affine)
        nib.save(atlas_seg_nii, atlas_seg_path)

        warped_atlas_seg_path = os.path.join(save_folder, f"warped_atlas_seg.nii.gz")

        # Warp the atlas seg using reg_resample and save the warped file
        # Warp the atlas seg given a pre-computed transformation (vel) and save it
        cmd = (
            f'{NIFTYREG_PATH}/reg_resample '
            f'-ref "{image_path}" '
            f'-flo "{atlas_seg_path}" '
            f'-trans "{cpp_path}" '
            f'-res "{warped_atlas_seg_path}" '
            f'-inter 0 '
            f'-voff '
            f'-omp {OMP}'
        )
        os.system(cmd)

        # Load the warped atlas seg volume
        warped_atlas_seg_nii = nib.load(warped_atlas_seg_path)
        warped_atlas_seg = warped_atlas_seg_nii.get_fdata().astype(np.uint8)

        # Create the background segmentation
        warped_atlas_seg_mask = np.zeros_like(warped_atlas_seg)
        warped_atlas_seg_mask[warped_atlas_seg > 0] = 1

        return warped_atlas_seg_path, warped_atlas_seg_mask


    elif RESAMPLE_METHOD == 1:
        warped_atlas_seg_l_paths = [os.path.join(save_folder, f"warped_atlas_seg_{l}.nii.gz") for l in range(num_class)]
        for l in range(num_class):
            print(l)
            atlas_seg_l = np.zeros_like(atlas_seg)
            atlas_seg_l[atlas_seg==l] = 1

            # smooth the atlas
            if SIGMA>0:
                atlas_seg_l = gaussian_filter(atlas_seg_l, sigma=SIGMA, order=0, mode='nearest')

            # save atlas seg as input for reg_resample
            atlas_seg_l_path = os.path.join(save_folder, f"atlas_seg_{l}.nii.gz")
            atlas_seg_l_nii = nib.Nifti1Image(atlas_seg_l, atlas_seg_nii.affine)
            nib.save(atlas_seg_l_nii, atlas_seg_l_path)

            # where should reg_resample save the warped files
            warped_atlas_seg_l_path = warped_atlas_seg_l_paths[l]


            # Warp the atlas seg given a pre-computed transformation (vel) and save it
            cmd = (
                f'{NIFTYREG_PATH}/reg_resample '
                f'-ref "{image_path}" '
                f'-flo "{atlas_seg_l_path}" '
                f'-trans "{cpp_path}" '
                f'-res "{warped_atlas_seg_l_path}" '
                f'-inter 1 '
                f'-voff '
                f'-omp {OMP}'
            )
            os.system(cmd)

        sum_warped_atlas_segs = None
        for l in range(num_class):
            # Load and return the warped atlas proba numpy array
            warped_altas_seg_l_nii = nib.load(warped_atlas_seg_l_paths[l])
            warped_altas_seg_l = warped_altas_seg_l_nii.get_fdata().astype(np.float32)  # H x W x D; is smoothed

            if l == 0:
                sum_warped_atlas_segs = np.zeros_like(warped_altas_seg_l)
            else:
                sum_warped_atlas_segs += warped_altas_seg_l


        # reg_resample pads all images with 0. Include these regions as background by setting them to 1 in the background segmentation
        # sum_warped_atlas_segs is 0 for pixels that were added by padding
        warped_altas_seg_0_nii = nib.load(warped_atlas_seg_l_paths[0])
        warped_altas_seg_0 = warped_altas_seg_0_nii.get_fdata().astype(np.float32)
        warped_altas_seg_0[sum_warped_atlas_segs == 0] = 1.  # H x W x D x C ; label 0 map is now 1 where no label was present (which is where reg_resample padded with 0)
        warped_altas_seg_0_nii = nib.Nifti1Image(warped_altas_seg_0, warped_altas_seg_0_nii.affine)
        nib.save(warped_altas_seg_0_nii, warped_atlas_seg_l_paths[0])

        sum_warped_atlas_segs[sum_warped_atlas_segs == 0] = 1  # Update the sum as well for normalization

        # get probabilistic atlas by normalizing across label dimension
        # additionally calculate the atlas mask (where background has the largest probability)
        for l in range(num_class):
            warped_altas_seg_l_nii = nib.load(warped_atlas_seg_l_paths[l])
            warped_altas_seg_l = warped_altas_seg_l_nii.get_fdata().astype(np.float32)  # H x W x D; is smoothed
            warped_altas_seg_normalized_l = warped_altas_seg_l / sum_warped_atlas_segs
            warped_altas_seg_normalized_l_nii = nib.Nifti1Image(warped_altas_seg_normalized_l, atlas_seg_nii.affine)
            nib.save(warped_altas_seg_normalized_l_nii, warped_atlas_seg_l_paths[l])

            if l == 0:
                warped_altas_seg_0 = warped_altas_seg_l
                warped_atlas_seg_mask = np.zeros_like(warped_altas_seg_0)
            else:
                warped_atlas_seg_mask[warped_altas_seg_l > warped_altas_seg_0] = 1

        return warped_atlas_seg_l_paths, warped_atlas_seg_mask