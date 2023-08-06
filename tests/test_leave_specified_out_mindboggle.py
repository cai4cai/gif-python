import os
import time

from src.multi_atlas.multi_atlas_segmentation import multi_atlas_segmentation
from glob import glob
import argparse

from src.utils.definitions import ROOT_DIR

# list of all atlases, including leave-out atlases
atlas_dir_list = [d for d in glob(ROOT_DIR+"/data/atlases/Mindboggle_atlases/*") if os.path.isdir(d)]

# check if argument --leaveout is provided, which can be used to leave out one or more atlases when predicting
parser = argparse.ArgumentParser()
parser.add_argument('--leaveout', action='extend', nargs='+', type=str, help='leave out this atlas when predicting')
args = parser.parse_args()

if args.leaveout:
    # check if all atlases to leave out are actually present
    for lo in args.leaveout:
        if not any([lo in d for d in atlas_dir_list]):
            raise ValueError(f"Atlas to leave out {lo} not found in atlas directory {os.path.dirname(atlas_dir_list[0])}")

    atlas_leaveout_list = [d for d in atlas_dir_list if any([lo in d for lo in args.leaveout])]  # leave out atlases
    atlas_remain_list = [d for d in atlas_dir_list if not any([lo in d for lo in args.leaveout])]  # remaining atlases

    print("Leaving out atlases: ", [os.path.basename(d) for d in atlas_leaveout_list])
    print("Atlases used for prediction: ", [os.path.basename(d) for d in atlas_remain_list])
else:
    raise ValueError("Please provide atlases to leave out with --leaveout argument")


# loop over left-out atlases, predict with all other atlases
for i in range(len(atlas_leaveout_list)):
    img_path = os.path.join(atlas_leaveout_list[i], 'orig_mni_aligned.nii.gz')
    mask_path = None  # these atlases don't have masks

    results_dir = ROOT_DIR+"/data/results/results_Mindboggle_atlases_leaveoneout/" + os.path.basename(atlas_leaveout_list[i])

    # assemble atlas paths dicts list, excluding the atlas we're leaving out
    atlas_paths_dicts_list = [{'name': os.path.basename(atlas_dir),
                               'img_path': os.path.join(atlas_dir, 'orig_mni_aligned.nii.gz'),
                               'seg_path': os.path.join(atlas_dir, 'labels_cleaned_mni_aligned.nii.gz')}
                              for atlas_dir in atlas_remain_list]

    time_0 = time.time()
    pred_atlas = multi_atlas_segmentation(
            img_path=img_path,
            mask_path=mask_path,
            atlas_paths_dicts_list=atlas_paths_dicts_list,
            structure_info_csv_path=os.path.join(os.path.dirname(atlas_remain_list[0]), 'structures_info.csv'),
            tissue_info_csv_path=os.path.join(os.path.dirname(atlas_remain_list[0]), 'tissues_info.csv'),
            save_dir=results_dir,
            )
    print("Total running time: ", time.time() - time_0, " seconds")

    seg_out_path = os.path.join(results_dir, "final_parcellation.nii.gz")
    print("Multi-atlas segmentation output saved to: ", seg_out_path)
    print("Visualizing segmentation output with itksnap...")
    os.system("itksnap -g " + img_path + " -s " + seg_out_path)

