import os
import time

from src.multi_atlas.inference import multi_atlas_segmentation

from glob import glob

NUM_CLASS=160

img_path = "./data/input/BraTS2021_00000/BraTS2021_00000_t1.nii.gz"
mask_path = "./data/input/BraTS2021_00000/BraTS2021_00000_inv-tumor-mask.nii.gz"
atlas_list = [d for d in glob("./data/GENFI_atlases/*") if os.path.isdir(d)]
atlas_pred_save_folder = "./data/results_GENFI_atlases"

time_0 = time.time()
pred_atlas = multi_atlas_segmentation(
        img_path=img_path,
        mask_path= mask_path,
        atlas_folder_list=atlas_list,
        num_class=NUM_CLASS,
        save_folder=atlas_pred_save_folder,
        only_affine=False,
    )
print("Total running time: ", time.time() - time_0, " seconds")

seg_out_path = os.path.join(atlas_pred_save_folder, "predicted_segmentation.nii.gz")
os.system("itksnap -g " + img_path + " -s " + seg_out_path)

