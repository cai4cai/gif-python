import argparse
import os
import subprocess
import numpy as np
import nibabel as nib
from glob import glob
from natsort import natsorted
from tqdm import tqdm

nib.Nifti1Header.quaternion_threshold = -np.finfo(np.float32).eps * 10  # CANDI dataset doesn't pass this check

def main():
    parser = argparse.ArgumentParser(description="Print (or, with --view, open in ITK-SNAP) the MNI pre-alignment of every NMM atlas.")
    parser.add_argument("--fixed-image", required=True, help="MNI template image")
    parser.add_argument("--atlas-dir", default="data/atlases/NMM_atlases", help="folder with one folder per atlas")
    parser.add_argument("--view", action="store_true", help="launch ITK-SNAP for each atlas (needs the itksnap executable)")
    args = parser.parse_args()

    fixed_image_path = args.fixed_image
    atlas_dir = args.atlas_dir
    images_to_align = natsorted(glob(os.path.join(atlas_dir, "*", "srr.nii*")))
    segmentations_to_align = natsorted(glob(os.path.join(atlas_dir, "*", "parcellation.nii.gz")))
    aligned_images = natsorted(glob(os.path.join(atlas_dir, "*", "srr_mni_aligned.nii.gz")))
    aligned_segmentations = natsorted(glob(os.path.join(atlas_dir, "*", "parcellation_mni_aligned.nii.gz")))

    assert (len(images_to_align) == len(segmentations_to_align) == len(aligned_images) == len(aligned_segmentations)), \
        (f"Expected the same number of images, segmentations and label maps, found {len(images_to_align)} original images, "
         f"{len(segmentations_to_align)} original segmentations, {len(aligned_images)} aligned images and "
         f"{len(aligned_segmentations)} aligned segmentations.")

    for moving_image_path, seg_path, img_mni, seg_mni in zip(tqdm(images_to_align), segmentations_to_align, aligned_images, aligned_segmentations):

        # check registration in itksnap
        command = ["itksnap", "-g", img_mni, "-o", fixed_image_path, "-s", seg_mni]
        print(" ".join(command))
        if args.view:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()


