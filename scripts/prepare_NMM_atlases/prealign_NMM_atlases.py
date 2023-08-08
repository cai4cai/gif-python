import os
import numpy as np
import nibabel as nib
from glob import glob
from natsort import natsorted
from tqdm import tqdm

nib.Nifti1Header.quaternion_threshold = -np.finfo(np.float32).eps * 10  # CANDI dataset doesn't pass this check

def main():

    fixed_image_path = "/mnt/nas1/fernando/reference/MNI_152_mri.nii.gz"
    images_to_align = natsorted(glob("../data/atlases/NMM_atlases/*/srr.nii*"))
    segmentations_to_align = natsorted(glob("../data/atlases/NMM_atlases/*/parcellation.nii.gz"))

    assert(len(images_to_align) == len(segmentations_to_align)), (("Expected the same number of images and "
                                                                   "segmentations, found ")
                                                                  + str(len(images_to_align)) + " images and "
                                                                  + str(len(segmentations_to_align)) + " segmentations.")

    for moving_image_path, seg_path in zip(tqdm(images_to_align), segmentations_to_align):
        # Perform registration with niftyreg
        output_dir = os.path.dirname(moving_image_path)

        sform_affine, niftyreg_resampled_img = get_affine_with_niftyreg(fixed_image_path, moving_image_path, output_dir, no_overwrite=True)
        registered_sform = get_registered_sform(moving_image_path, sform_affine)
        sform_registered_output_img_path = update_sform(moving_image_path, registered_sform, output_dir)
        sform_registered_output_seg_path = update_sform(seg_path, registered_sform, output_dir)

        # # check registration in itksnap
        # os.system(f"itksnap -g {sform_registered_output_img_path} "
        #           f" -o {fixed_image_path} {niftyreg_resampled_img} "
        #           f" -s {sform_registered_output_seg_path}")


def get_affine_with_niftyreg(fixed_image_path, moving_image_path, output_dir, no_overwrite=False):
    """
    Perform affine registration with niftyreg
    :param fixed_image_path: path to the fixed image
    :param moving_image_path: path to the moving image
    :param output_dir: path to the output directory
    :param no_overwrite: if True, do not overwrite the output files if they already exist
    :return: the affine matrix from sform of the fixed image to the sform of the moving image, as a numpy array
        That is, if the returned affine is A, then the image data of the moving image (m_i) should be transformed to the
        image data for the fixed image (f_i) as follows:
        f_i = inv(sform_f_i) @ inv(A) @ sform_m_i @ m_i


    """
    resampled_img_output_path = os.path.join(output_dir, os.path.basename(moving_image_path).replace(".nii.gz", ".nii").replace(".nii", "_mni_resampled.nii.gz"))
    output_affine_path = os.path.join(output_dir, "affine_from_mni_to_" + os.path.basename(moving_image_path).replace(".nii.gz", ".nii").replace(".nii", "") + ".txt")

    if no_overwrite and os.path.exists(resampled_img_output_path) and os.path.exists(output_affine_path):
        print("Affine registration already performed, skipping.")
    else:
        os.system(f"reg_aladin -ref " + fixed_image_path +
                  " -flo " + moving_image_path +
                  " -res " + resampled_img_output_path +
                  " -aff " + output_affine_path +
                  " -maxit 5" +
                  " -ln 3" +
                  " -lp 2" +
                  " -speeeeed" +
                  " -voff"
                  )

    # read the affine matrix
    output_affine = np.loadtxt(output_affine_path)

    return output_affine, resampled_img_output_path


def get_registered_sform(moving_image_path, sform_affine):
    """
    Get the sform of the original moving image pixel coordinates to the world coordinates when registered to the fixed
    image
    :param moving_image_path: path to the moving image
    :param sform_affine: the affine matrix from the sform of the fixed image to the sform of the moving image
    :return: the sform of the original moving image pixel coordinates to the world coordinates when registered to the
        fixed image
    """

    # get the original sform from the header
    sform_old = nib.load(moving_image_path).affine

    # invert the affine matrix
    affine_inv = np.linalg.inv(sform_affine)

    registered_sform = affine_inv @ sform_old

    return registered_sform


def update_sform(moving_image_path, registered_sform, output_dir):
    """
    Update the sform of the moving image to the registered sform
    :param moving_image_path: path to the moving image
    :param registered_sform: the sform of the original moving image pixel coordinates to the world coordinates when
        registered to the fixed image
    :param output_dir: path to the output directory
    :return: None
    """
    nii = nib.load(moving_image_path)

    # update the sform
    # save the new affine matrix to the header
    nii.set_qform(registered_sform)  # to avoid ambiguity, set the qform as well (note that the qform does not handle shearing, so it can still end up being different from the sform)
    nii.set_sform(registered_sform)

    nii.header['sform_code'] = 4  # mni aligned
    nii.header['qform_code'] = 0  # unknown

    # save the image
    sform_registered_output_path = os.path.join(output_dir, os.path.basename(moving_image_path).replace(".nii.gz", ".nii").replace(".nii", "_mni_aligned.nii.gz"))
    nib.save(nii, sform_registered_output_path)

    return sform_registered_output_path


if __name__ == "__main__":
    main()


