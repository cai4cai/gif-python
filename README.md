# Python implementation of the Geodesic Information Flows (GIF) algorithm
References
- Paper: [Geodesic Information Flows: Spatially-Variant Graphs and Their Application to Segmentation and Fusion ](https://pubmed.ncbi.nlm.nih.gov/25879909/)
- Reference c++ implementation: https://github.com/KCL-BMEIS/gif

## Steps performed by the algorithm
1. Register all atlas images to the image to be segmented (target image) and resample the atlas images into
the target image space. The registration is performed using the [NiftyReg](https://github.com/KCL-BMEIS/niftyreg) library. First, an affine 
registration is performed using reg_aladin, then a non-linear registration is performed using reg_f3d.
2. Resample the atlas segmentations into the space of the target image using the transformation computed in step 1. This 
is done using reg_resample from the [NiftyReg](https://github.com/KCL-BMEIS/niftyreg) library.
3. Compute weights that indicate the similarity between the target image and each atlas image. The weights are based on
the Local Normalized Cross Correlation (LNCC) between the target image and each atlas image.
4. Fuse the resampled atlas segmentations using the weights computed in step 3 to obtain a probability map of the 
segmentation of the target image for each label present in the atlas segmentations.
5. Using an input mapping between the labels and a much smaller number of tissue classes (gray matter, white matter, 
etc.), distribute the label probabilities to the tissue classes to create a tissues segmentation prior.
6. Use the tissue segmentation prior and the target image as input to an Expectation Maximization algorithm to obtain
the final tissue segmentation of the target image. This is done using the seg_EM binary from the 
[NiftySeg](https://github.com/KCL-BMEIS/niftyseg) library.
7. Create the final segmentation of the target image by assigning each voxel to the label with the highest probability
that matches the tissue segmentation obtained in step 6 according to the input mapping between labels and tissue
8. classes.

## System requirements
### Hardware requirements

The code has been tested with the configuration:
* Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz

### Software requirements
* Python 3.8 or higher
* [NiftyReg](https://github.com/KCL-BMEIS/niftyreg)
* [NiftySeg](https://github.com/KCL-BMEIS/niftyseg)

#### Libraries
The following python libraries are required:
* numpy
* nibabel
* scipy
* pandas
* numba

They can be installed with pip:
```pip install numpy nibabel scipy pandas numba```

## How to use
A test script is provided here: [run_multi_atlas_segmentation_downsampled_test.py](run_multi_atlas_segmentation_downsampled_test.py)

6 arguments need to be defined:
* ```img_path```: path to the image to be segmented
* ```mask_path```: path to the mask of the image to be segmented (1 for the region of interest, 0 for the background)
* ```atlas_dir_list```: list of paths to the atlas folder (each atlas folder contains the atlas image and the atlas segmentation)
* ```results_dir```: path to the folder where the results will be saved
* ```structure_info_csv_path```: path to the csv file containing labels numbers, label names and the mapping between
the labels in the atlas segmentations to the tissue classes, for example: 8,Right Accumbens Area,"[3, 4]"
* ```tissue_info_csv_path```: path to the csv file containing labels and names of the tissue classes: for example: 
3, White Matter

The script will create the following output in the ```results_dir```:

* ```<atlas_name>``` folder for each atlas in ```atlas_dir_list```. Each folder contains the following:
    * ```warped_atlas_image.nii.gz``` The atlas image resampled into the space of the target image
    * ```warped_atlas_seg.nii.gz``` The atlas segmentation resampled into the space of the target image
    * ```weights.nii.gz``` The weights computed for each atlas image
    * some other temporary files used or created by NiftyReg
* ```multi_atlas_tissue_prior.nii.gz``` The tissue segmentation prior
* ```multi_atlas_tissue_seg.nii.gz``` The final tissue segmentation of the target image
* ```final_parcellation.nii.gz``` The final parcellation (multi-atlas segmentation) of the target image

## Options
Multiprocessing options, parameters for Niftyreg and Niftyseg, and other options
can be modified in the [definitions.py](src/utils/definitions.py) file.
