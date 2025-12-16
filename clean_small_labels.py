import nibabel as nib
import os
import numpy as np
from glob import glob

labels_to_replace = [1001, 1032, 1033, 2001, 2032, 2033]

p_in_all = glob("data/Mindboggle101/*/labels.DKT31.manual+aseg.nii.gz")
p_out_all = [p.replace(".nii.gz", "_cleaned.nii.gz") for p in p_in_all]

def get_neighbors(coords):
    x, y, z = coords
    
    # Define offsets for all possible neighboring coordinates
    offsets = np.array([
        [-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
        [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
        [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
        [0, -1, -1], [0, -1, 0], [0, -1, 1],
        [0, 0, -1], [0, 0, 1],
        [0, 1, -1], [0, 1, 0], [0, 1, 1],
        [1, -1, -1], [1, -1, 0], [1, -1, 1],
        [1, 0, -1], [1, 0, 0], [1, 0, 1],
        [1, 1, -1], [1, 1, 0], [1, 1, 1]
    ])
    
    # Add the offsets to the input coordinates
    neighbor_coords = coords + offsets
    
    return neighbor_coords

for p_in, p_out in list(zip(p_in_all, p_out_all)):
    print(p_in)
    nii = nib.load(p_in)
    data = nii.get_fdata()

    data_out = np.copy(data)
    
    for l_in in labels_to_replace:

        locs = np.where(data==l_in)

        for i,j,k in zip(*locs):
            neighbor_vals = []
            #print("voxel: ", i,j,k)
            neighbors = get_neighbors([i,j,k])
            #print(neighbors)

            for n in neighbors:
                neighbor_val = data[n[0], n[1], n[2]]
                if neighbor_val != l_in:
                    neighbor_vals.append(neighbor_val)

            #print("neighbor label counts: ")
            if len(neighbor_vals) == 0: raise Exception("no neighbors found...")
            
            labels, counts = np.unique(np.array(neighbor_vals), return_counts=True)
            #print(labels, counts)

            max_counts_idx = np.argmax(counts)

            max_label = labels[max_counts_idx]

            assert(max_label not in labels_to_replace), f"{max_label=} in {labels_to_replace=}"

            #print("max_label", max_label)

            print(f"change {l_in} to {max_label}")
            data_out[i, j, k] = max_label

        
    print("number_of_changed_voxels", np.sum(data_out != data))

    nii_out = nib.Nifti1Image(dataobj=data_out, affine=nii.affine)
    nib.save(nii_out, filename=p_out)