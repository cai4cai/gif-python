import os
import time

from src.multi_atlas.multi_atlas_segmentation import multi_atlas_segmentation
from glob import glob
from natsort import natsorted

from src.utils.definitions import ROOT_DIR

import argparse

# read arguments from command line
parser = argparse.ArgumentParser(description='Run multi-atlas segmentation on UPENN-GBM dataset')
# read --taskid argument from command line
parser.add_argument('--taskid', type=int, help='taskid', required=True)
args = parser.parse_args()

taskid = args.taskid


img_paths = natsorted(glob(os.path.join(ROOT_DIR, "data", "input", "OpenNeuro", "openneurocoll_*", "openneurocoll_*.nii.gz")))

assert(len(img_paths) > 0), f"Did not find any images"
print(f"Found {len(img_paths)} images")
print(f"Running taskid {taskid} out of {len(img_paths)}, img_path = {img_paths[taskid]}")

img_path = img_paths[taskid]
mask_path = None

atlas_dir_list = [d for d in glob(ROOT_DIR+"/data/atlases/Mindboggle_atlases/*") if os.path.isdir(d)]
results_dir = os.path.join(ROOT_DIR, "data/results/results_Mindboggle_on_OpenNeuro",  os.path.basename(os.path.dirname(img_path)))

atlas_paths_dicts_list = [{'name': os.path.basename(atlas_dir),
                           'img_path': os.path.join(atlas_dir, 'orig_mni_aligned.nii.gz'),
                           'seg_path': os.path.join(atlas_dir, 'labels_cleaned_mni_aligned.nii.gz')}
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

