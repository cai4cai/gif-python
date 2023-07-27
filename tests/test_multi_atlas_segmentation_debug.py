import os
import time

from src.multi_atlas.multi_atlas_segmentation import multi_atlas_segmentation
from glob import glob


root_dir = "."
img_path = os.path.join(root_dir, "data/input/BraTS2021_00000/BraTS2021_00000_t1.nii.gz")
mask_path = os.path.join(root_dir, "data/input/BraTS2021_00000/BraTS2021_00000_inv-tumor-mask.nii.gz")
atlas_dir_list = [d for d in glob(os.path.join(root_dir, "data/atlases/GENFI_atlases_debug/*")) if os.path.isdir(d)]
results_dir = os.path.join(root_dir, "data/results/results_GENFI_atlases_debug")

print("img_path = ", os.path.abspath(img_path))

time_0 = time.time()
pred_atlas = multi_atlas_segmentation(
        img_path=img_path,
        mask_path=mask_path,
        atlas_dir_list=atlas_dir_list,
        structure_info_csv_path=os.path.join(os.path.dirname(atlas_dir_list[0]), 'structures_info.csv'),
        tissue_info_csv_path=os.path.join(os.path.dirname(atlas_dir_list[0]), 'tissues_info.csv'),
        save_dir=results_dir,
        )
print("Total running time: ", time.time() - time_0, " seconds")

seg_out_path = os.path.join(results_dir, "final_parcellation.nii.gz")
print("Multi-atlas segmentation output saved to: ", seg_out_path)
print("Visualizing segmentation output with itksnap...")
os.system("itksnap -g " + img_path + " -s " + seg_out_path)

