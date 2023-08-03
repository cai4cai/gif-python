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
    aligned_images = natsorted(glob("../data/atlases/NMM_atlases/*/srr_mni_aligned.nii.gz"))
    aligned_segmentations = natsorted(glob("../data/atlases/NMM_atlases/*/parcellation_mni_aligned.nii.gz"))

    assert (len(images_to_align) == len(segmentations_to_align) == len(aligned_images) == len(aligned_segmentations)), \
        (f"Expected the same number of images, segmentations and label maps, found {len(images_to_align)} original images, "
         f"{len(segmentations_to_align)} original segmentations, {len(aligned_images)} aligned images and "
         f"{len(aligned_segmentations)} aligned segmentations.")

    for moving_image_path, seg_path, img_mni, seg_mni in zip(tqdm(images_to_align), segmentations_to_align, aligned_images, aligned_segmentations):

        # check registration in itksnap
        os.system(f"itksnap -g {img_mni} "
                  f" -o {fixed_image_path}"
                  f" -s {seg_mni}")


if __name__ == "__main__":
    main()


