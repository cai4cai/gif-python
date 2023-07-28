import os
from glob import glob
from natsort import natsorted
import shutil
import pandas as pd

nmm_ds_path = "/mnt/nas1/Datasets/NMM_BrainParc/2022-11-20_Neuromorphometrics-AcademicDataset"

out_atlas_dir = "/home/aaron/Dropbox/KCL/Projects/gif-python/data/atlases/NMM_atlases"

# create atlas dictionary list
atlas_paths_dicts_list = []

sub_ds = [d for d in natsorted(glob(os.path.join(nmm_ds_path, "*"))) if os.path.isdir(d)]

first_dirs = []
for sub in sub_ds:
    atlas_dirs = [d for d in natsorted(glob(os.path.join(sub, "*"))) if os.path.isdir(d)]
    first_dirs.append(atlas_dirs[0])

for d in first_dirs:
    print(d)
    files = glob(os.path.join(d, "*"))

    for f in files:
        print(os.path.basename(f))

    assert(len(files) == 3), "Expected 3 files in each directory, found " + str(len(files)) + " files."

    # check that the files are named correctly

    # one file needs to end on "_seg.nii.gz"
    assert(any([f.endswith("_seg.nii.gz") for f in files])), "Expected one file to end on '_seg.nii.gz', found none."

    # one file needs to end on "_LabelMap.xml"
    assert(any([f.endswith("_LabelMap.xml") for f in files])), "Expected one file to end on '_LabelMap.xml', found none."

    # the remaining file needs to end on ".nii.gz"
    assert(any([f.endswith(".nii.gz") and not f.endswith("_seg.nii.gz") for f in files])), "Expected one file to end on '.nii.gz', found none."

    # get the sub-dataset name from the path
    sub_ds_name = d.split("/")[-2]

    # assemble the atlas name from the file names
    name = ""
    for char1, char2, char3 in zip(*[os.path.basename(f) for f in files]):
        if char1 == char2 and char2 == char3:
            name += char1
        else:
            break

    atlas_name = sub_ds_name.split("_")[0] + "_" + name
    print("atlas name = ", atlas_name)

    # remove fileparts before "data" from the atlas_dir


    # copy the files to the atlas directory while and add dictionary to list
    atlas_dir = os.path.join(out_atlas_dir, sub_ds_name.split("_")[0] + "_" + name)
    relative_atlas_dir = os.path.join(".", "data", atlas_dir.split("data"+os.sep)[1])
    os.makedirs(atlas_dir, exist_ok=True)
    new_dict = {"name": atlas_name}
    for f in files:
        shutil.copy(f, atlas_dir)
        if f.endswith("_seg.nii.gz"):
            new_dict["seg_path"] = os.path.join(relative_atlas_dir, os.path.basename(f))
        elif f.endswith("_LabelMap.xml"):
            # no need to copy the xml file
            pass
        else:
            new_dict["img_path"] = os.path.join(relative_atlas_dir, os.path.basename(f))

    atlas_paths_dicts_list.append(new_dict)

    print("atlas directory = ", atlas_dir)
    print("atlas dictionary = ", new_dict)

    break

pd.DataFrame(atlas_paths_dicts_list).to_csv(os.path.join(out_atlas_dir, "atlas_paths.csv"), index=False)






