import os
from datetime import time

from src.multi_atlas.inference import multi_atlas_segmentation
import nibabel as nib
import numpy as np
from src.utils.definitions import BE, LE, LP, GRID_SPACING
from glob import glob

from src.utils.definitions import NUM_CLASS

img_path = "./data/input/BraTS2021_00000/BraTS2021_00000_t1.nii.gz"
mask_path = "./data/input/BraTS2021_00000/BraTS2021_00000_inv-tumor-mask.nii.gz"
atlas_list = [d for d in glob("./data/GENFI_atlases_small_numclass/*") if os.path.isdir(d)]
atlas_pred_save_folder = "./data/results_GENFI_atlases_small_numclass"

MERGING_MULTI_ATLAS = 'GIF'

time_0 = time.time()
pred_atlas = multi_atlas_segmentation(
        img_path=img_path,
        mask_path= mask_path,
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
time_1 = time.time()
print("Total running time: ", time_1 - time_0, " seconds")