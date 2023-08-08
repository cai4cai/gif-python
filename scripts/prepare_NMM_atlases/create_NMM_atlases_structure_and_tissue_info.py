import os
import warnings
from glob import glob
from natsort import natsorted
import pandas as pd
import xmltodict
import nibabel as nib
import numpy as np

warnings.simplefilter('always', UserWarning)

nmm_ds_path = "/mnt/nas1/Datasets/NMM_BrainParc/NMM_BrainParc_clean"

out_nmm_atlas_dir = "/home/aaron/Dropbox/KCL/Projects/gif-python/data/atlases/NMM_atlases"

# get GENFI labels, since they have the tissue assignments (not available in NMM datasets)
df_GENFI = pd.read_csv("/home/aaron/Dropbox/KCL/Projects/gif-python/data/atlases/GENFI_atlases/structures_info.csv")
GENFI_descs = df_GENFI['name'].values

images = [i for i in natsorted(glob(os.path.join(nmm_ds_path, "images", "*_0000.nii.gz")))]
segmentations = [i for i in natsorted(glob(os.path.join(nmm_ds_path, "labels", "*seg.nii.gz")))]
label_maps = [i for i in natsorted(glob(os.path.join(nmm_ds_path, "xml", "*.xml")))]

assert(len(images) == len(segmentations) == len(label_maps)), (("Expected the same number of images, segmentations and "
                                                               "label maps, found ")
                                                               + str(len(images)) + " images, "
                                                               + str(len(segmentations)) + " segmentations and "
                                                               + str(len(label_maps)) + " label maps.")

skipped_atlases = []
nmm_label_dicts_list_with_tissues = []
for i, (img_p, seg_p, xml_p) in enumerate(zip(images, segmentations, label_maps)):
    print(img_p)
    print(seg_p)
    print(xml_p)

    # get the sub-dataset name from the path
    sub_ds_name = os.path.basename(img_p).split("_")[0]

    # get the atlas name from the path
    atlas_name = os.path.basename(img_p).split("_0000")[0]
    print("atlas name = ", atlas_name)

    # get the set of labels from the _seg.nii.gz file
    seg = nib.load(seg_p).get_fdata()
    nib.Nifti1Header.quaternion_threshold = -np.finfo(np.float32).eps * 10  # CANDI dataset doesn't pass this check
    labels = np.unique(seg)

    print("Found a total of ", len(labels), " labels in the segmentation.")
    print("Labels: ", labels)

    with open(xml_p, "r") as f:
        xml_string = f.read()

    xml_dict = xmltodict.parse(xml_string)
    nmm_label_dicts_list = [{"label": int(d['Number']), "name": d['Name']} for d in xml_dict['LabelList']['Label']]
    nmm_label_dicts_list = sorted(nmm_label_dicts_list, key=lambda k: k['label'])

    # get the tissue assignments from the GENFI dataset
    nb_matches = 0
    for nmm_d in nmm_label_dicts_list:
        desc2 = nmm_d['name']
        label = nmm_d['label']
        for desc in GENFI_descs:
            if desc.replace(" ", "") == desc2.replace(" ", ""):
                # print(desc)
                nb_matches += 1

                # get the tissue types from the GENFI dataset
                tissue = df_GENFI[df_GENFI['name'] == desc]['tissues'].iloc[0]
                nmm_d['tissues'] = tissue

                # add the dictionary to the list if it is not already in there
                if nmm_d not in nmm_label_dicts_list_with_tissues:
                    nmm_label_dicts_list_with_tissues.append(nmm_d)
                break
        else:
            pass
            if label in labels:  # check that the current not-found label is actually present in the segmentation or just mentioned in the xml file
                print("----------------------------------- Not found:", desc2, " with label ", label)
                # assign tissue manually
                if "Lesion" in desc2:
                    warnings.warn("Lesion present, skip this atlas...")
                    if atlas_name not in skipped_atlases:
                        skipped_atlases.append(atlas_name)
                    break
                elif "Unlabeled" in desc2:
                    print("assign this label to all tissues")
                    nmm_d['tissues'] = [0, 1, 2, 3, 4, 5]
                elif "Vitamin E" in desc2:
                    print("assign this label to background")
                    nmm_d['tissues'] = [0]
                elif "White Matter" in desc2:
                    print("assign white matter to this label")
                    nmm_d['tissues'] = [3]
                elif "CSF" in desc2:
                    nb_matches += 1
                    print("Matched manually with 'Non-ventricular CSF' label ... assign CSF to this label")
                    nmm_d['tissues'] = [1]
                elif "Basal Forebrain" in desc2:
                    nb_matches += 1
                    print("Matched manually with 'Basal Forebrain' label ... assign Deep Grey Matter to this label")
                    nmm_d['tissues'] = [4]
                elif "icc" in desc2:
                    print("assign CSF to this label")
                    nmm_d['tissues'] = [1]
                else:
                    raise ValueError("Unknown tissue type: " + desc2 + " with label " + str(label))

                    # # write a warning with warnings library
                    # warnings.warn("Unknown tissue type: " + desc2 + " with label " + str(label))

                if nmm_d not in nmm_label_dicts_list_with_tissues:
                    nmm_label_dicts_list_with_tissues.append(nmm_d)

    print(f"{nb_matches=}")
    print("nmm_label_dicts_list_with_tissues = ", nmm_label_dicts_list_with_tissues)

# add background label manually
nmm_label_dicts_list_with_tissues.append({"label": 0, "name": "Background", "tissues": [0]})

# convert the list of dictionaries to a dataframe
df = pd.DataFrame(nmm_label_dicts_list_with_tissues)

# sort by label
df = df.sort_values(by=['label'])

# save the dataframe to a csv file
df.to_csv(os.path.join(out_nmm_atlas_dir, "structures_info.csv"), index=False)

# skipped atlases
print("skipped atlases", skipped_atlases)
