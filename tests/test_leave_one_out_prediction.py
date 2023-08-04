import os
import time

from src.multi_atlas.multi_atlas_segmentation import multi_atlas_segmentation
from glob import glob

atlas_dir_list = [d for d in glob("./data/atlases/NMM_atlases/*") if os.path.isdir(d)]

# loop over all atlases, predict with all other atlases
for i in range(len(atlas_dir_list)):
    img_path = os.path.join(atlas_dir_list[i], 'srr_mni_aligned.nii.gz')
    mask_path = None  # these atlases don't have masks

    results_dir = "./data/results/results_NMM_atlases_leaveoneout/" + os.path.basename(atlas_dir_list[i])

    # assemble atlas paths dicts list, excluding the atlas we're leaving out
    atlas_paths_dicts_list = [{'name': os.path.basename(atlas_dir),
                               'img_path': os.path.join(atlas_dir, 'srr_mni_aligned.nii.gz'),
                               'seg_path': os.path.join(atlas_dir, 'parcellation_mni_aligned.nii.gz')}
                              for atlas_dir in atlas_dir_list if atlas_dir != atlas_dir_list[i]]

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

