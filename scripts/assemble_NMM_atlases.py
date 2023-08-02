import os
from glob import glob
from natsort import natsorted
import shutil
import pandas as pd

nmm_ds_path = "/mnt/nas1/Datasets/NMM_BrainParc/NMM_BrainParc_clean"

out_atlas_dir = "/home/aaron/Dropbox/KCL/Projects/gif-python/data/atlases/NMM_atlases"

images = [i for i in natsorted(glob(os.path.join(nmm_ds_path, "images", "*_0000.nii.gz")))]
segmentations = [i for i in natsorted(glob(os.path.join(nmm_ds_path, "labels", "*seg.nii.gz")))]
label_maps = [i for i in natsorted(glob(os.path.join(nmm_ds_path, "xml", "*.xml")))]

assert(len(images) == len(segmentations) == len(label_maps)), (("Expected the same number of images, segmentations and "
                                                               "label maps, found ")
                                                               + str(len(images)) + " images, "
                                                               + str(len(segmentations)) + " segmentations and "
                                                               + str(len(label_maps)) + " label maps.")

# create atlas dictionary list
atlas_paths_dicts_list = []

sub_ds = [d for d in natsorted(glob(os.path.join(nmm_ds_path, "*"))) if os.path.isdir(d)]


for i, (img_p, seg_p, xml_p) in enumerate(zip(images, segmentations, label_maps)):
    print(img_p)
    print(seg_p)
    print(xml_p)

    # get the sub-dataset name from the path
    sub_ds_name = os.path.basename(img_p).split("_")[0]

    # get the atlas name from the path
    atlas_name = os.path.basename(img_p).split("_0000")[0]
    print("atlas name = ", atlas_name)

    # copy the files to the atlas directory while and add dictionary to list
    atlas_dir = os.path.join(out_atlas_dir, atlas_name)
    relative_atlas_dir = os.path.join(".", "data", atlas_dir.split("data"+os.sep)[1])
    os.makedirs(atlas_dir, exist_ok=True)
    new_dict = {"name": atlas_name}

    # copy the segmentation
    dst_seg = os.path.join(atlas_dir, "parcellation.nii.gz")
    new_dict["seg_path"] = dst_seg
    shutil.copy(seg_p, dst_seg)

    # copy the image
    dst_img = os.path.join(atlas_dir, "srr.nii.gz")
    new_dict["img_path"] = dst_img
    shutil.copy(img_p, dst_img)

    atlas_paths_dicts_list.append(new_dict)

    print("atlas directory = ", atlas_dir)
    print("atlas dictionary = ", new_dict)


pd.DataFrame(atlas_paths_dicts_list).to_csv(os.path.join(out_atlas_dir, "atlas_paths.csv"), index=False)






