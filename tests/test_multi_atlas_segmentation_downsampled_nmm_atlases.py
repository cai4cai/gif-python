import os
import time

from src.multi_atlas.multi_atlas_segmentation import multi_atlas_segmentation
from glob import glob

from src.utils.definitions import ROOT_DIR

img_path = ROOT_DIR+"/data/input/BraTS2021_00000_downsampled/BraTS2021_00000_t1.nii.gz"
mask_path = ROOT_DIR+"/data/input/BraTS2021_00000_downsampled/BraTS2021_00000_inv-tumor-mask.nii.gz"
atlas_dir_list = [d for d in glob(ROOT_DIR+"/data/atlases/NMM_atlases/*") if os.path.isdir(d)]
results_dir = ROOT_DIR+"/data/results/results_NMM_atlases_downsampledInput"

atlas_paths_dicts_list = [{'name': os.path.basename(atlas_dir),
                           'img_path': os.path.join(atlas_dir, 'srr_mni_aligned.nii.gz'),
                           'seg_path': os.path.join(atlas_dir, 'parcellation_mni_aligned.nii.gz')}
                          for atlas_dir in atlas_dir_list]

time_0 = time.time()
pred_atlas = multi_atlas_segmentation(
        img_path=img_path,
        mask_path=mask_path,
        atlas_paths_dicts_list=atlas_paths_dicts_list,
        structure_info_csv_path=os.path.join(os.path.dirname(atlas_dir_list[0]), 'structures_info.csv'),
        tissue_info_csv_path=os.path.join(os.path.dirname(atlas_dir_list[0]), 'tissues_info.csv'),
        save_dir=results_dir,
        )
print("Total running time: ", time.time() - time_0, " seconds")

seg_out_path = os.path.join(results_dir, "final_parcellation.nii.gz")
print("Multi-atlas segmentation output saved to: ", seg_out_path)
print("Visualizing segmentation output with itksnap...")
os.system("itksnap -g " + img_path + " -s " + seg_out_path)

