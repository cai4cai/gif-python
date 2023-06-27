import os

from src.multi_atlas.inference import multi_atlas_segmentation
import nibabel as nib
import numpy as np
from src.utils.definitions import BE, LE, LP, GRID_SPACING
from glob import glob

NUM_CLASS=160

input_path = "./data/input/BraTS2021_00000/BraTS2021_00000_t1.nii.gz"
mask_path = "./data/input/BraTS2021_00000/BraTS2021_00000_inv-tumor-mask.nii.gz"
atlas_list = glob("./data/GENFI_atlases/*")
atlas_pred_save_folder = "./data/results_GENFI_atlases"

img_nii = nib.load(input_path)
img = img_nii.get_fdata().astype(np.float32)
mask_nii = nib.load(mask_path)
mask = mask_nii.get_fdata().astype(np.uint8)


MERGING_MULTI_ATLAS = 'GIF'

pred_proba_atlas = multi_atlas_segmentation(
        img_nii,
        mask_nii,
        atlas_folder_list=atlas_list,
        num_class=NUM_CLASS,
        grid_spacing=GRID_SPACING,
        be=BE,
        le=LE,
        lp=LP,
        save_folder=atlas_pred_save_folder,
        only_affine=False,
        merging_method=MERGING_MULTI_ATLAS,
        reuse_existing_pred=False,
        force_recompute_heat_kernels=False,
    )
    
predicted_segmentation = np.argmax(pred_proba_atlas, axis=3).astype(np.uint8)*mask_nii.get_fdata()

predicted_segmentation_nii = nib.Nifti1Image(predicted_segmentation, img_nii.affine)
nib.save(predicted_segmentation_nii, os.path.join(atlas_pred_save_folder, "predicted_segmentation.nii.gz"))

