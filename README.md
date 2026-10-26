# gif-python

Python implementation of the Geodesic Information Flows (GIF) multi-atlas segmentation. It drives the
[NiftyReg](https://github.com/KCL-BMEIS/niftyreg) and [NiftySeg](https://github.com/KCL-BMEIS/niftyseg)
command line tools and does the label fusion in Python.

References
- Paper: [Geodesic Information Flows: Spatially-Variant Graphs and Their Application to Segmentation and Fusion](https://pubmed.ncbi.nlm.nih.gov/25879909/)
- Reference C++ implementation: https://github.com/KCL-BMEIS/gif

The distribution is called `pygif`; the code is imported as `src` (`from src.multi_atlas.multi_atlas_segmentation import multi_atlas_segmentation`).

## Installation

Python 3.9 or newer, plus the NiftyReg and NiftySeg executables (`reg_aladin`, `reg_f3d`, `reg_resample`, `seg_EM`);
see [INSTALL.md](INSTALL.md) for how to build them.

```bash
pip install git+https://github.com/cai4cai/gif-python.git     # or, from a checkout: pip install .
gif-python --help
```

The executables are found through the environment variables `NIFTYREG_PATH` and `NIFTYSEG_PATH` (the directories that
contain them), then on `PATH`, then in `/usr/local/bin`.

## Quickstart

Segment `image.nii.gz` with the atlases in `atlases/` (one folder per atlas with an image and a segmentation):

```bash
gif-python segment --img image.nii.gz --atlas-dir atlases \
    --atlas-img img.nii.gz --atlas-seg seg.nii.gz \
    --structure-info atlases/structures_info.csv --tissue-info atlases/tissues_info.csv \
    --out results/image
```

Results are written to `results/image/`; `final_parcellation.nii.gz` is the segmentation. All options:
`gif-python segment --help`. Parameters of the pipeline are changed with `--config NAME=VALUE`
(list them with `gif-python segment --print-config`), see [docs/tuning-registration.md](docs/tuning-registration.md).

From Python:

```python
from src.multi_atlas.multi_atlas_segmentation import multi_atlas_segmentation
from src.utils.config import GifConfig

multi_atlas_segmentation(
    img_path="image.nii.gz",
    mask_path=None,
    atlas_paths_dicts_list=[{"name": "a1", "img_path": "a1/img.nii.gz", "seg_path": "a1/seg.nii.gz"}],
    structure_info_csv_path="structures_info.csv",
    tissue_info_csv_path="tissues_info.csv",
    save_dir="results/image",
    config=GifConfig(reg_f3d_lncc=7.0, omp=4),  # optional
)
```

## Mindboggle-101 quickstart

```bash
gif-python fetch-mindboggle --clean        # downloads and verifies the 101 volumes (about 3.3 GB), cleans the labels
gif-python segment \
    --img data/atlases/Mindboggle101/OASIS-TRT-20-1/t1weighted_brain.nii.gz \
    --atlas-dir data/atlases/Mindboggle101 --exclude-atlas OASIS-TRT-20-1 \
    --out outputs/OASIS-TRT-20-1
```

The label tables (`structures_info.csv`, `tissues_info.csv`, with all labels of the data set) and an ITK-SNAP label
file are shipped with the package and copied to the atlas folder. Details, the label cleaning and the open questions are in
[docs/mindboggle101.md](docs/mindboggle101.md).

## Commands

| Command | Purpose |
|---|---|
| `gif-python segment` | multi-atlas segmentation of one image |
| `gif-python clean-labels` | replace sparse labels by the most frequent neighbouring label |
| `gif-python make-structure-info` | create `structures_info.csv` / `tissues_info.csv` from an atlas segmentation and a FreeSurfer colour table |
| `gif-python fetch-mindboggle` | download and verify the Mindboggle-101 volumes |

The former top-level scripts (`run_registration_parcellation_basic.py`, `clean_small_labels.py`, ...) still work and forward to these commands.

## GPU backend (optional)

`gif-python segment --backend gpu` (or `GifConfig(backend="gpu")`) runs the whole pipeline - registration, fusion and the
tissue EM - on a CUDA GPU in a few seconds instead of minutes, with the same inputs and output files. It needs the `gif_gpu`
program built from [`native/`](native/README.md) (CMake, CUDA toolkit); the default backend is still `cpu`. The results are
statistically equivalent to the CPU backend, not bit-identical (the registration is chaotic); masks are not supported.
See [native/README.md](native/README.md) for the build, the limits and the validation against the CPU backend.

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
classes.

## Inputs and outputs

* `structures_info.csv`: label number, label name and the list of tissues the structure can belong to, for example
  `8,Right Accumbens Area,"[3, 4]"`. Every label that occurs in an atlas segmentation must be listed.
* `tissues_info.csv`: tissue number and name, for example `3,White Matter`.
* Optional masks of the target and the atlases (1 inside the region of interest, 0 outside).

Written to the output folder:

* `registered_atlases/<atlas name>/warped_atlas_seg.nii.gz`: the atlas segmentation resampled into the target space
  (with `--keep-registration-files` also the affine, the control point grid and the warped atlas image).
* `multi_atlas_proba_seg_argmax.nii.gz`: label with the highest fused weight in each voxel.
* `multi_atlas_tissue_prior.nii.gz`: the tissue prior (4D).
* `multi_atlas_tissue_seg.nii.gz`: output of the tissue EM (4D tissue probabilities).
* `final_parcellation.nii.gz`: the final parcellation.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest                  # unit tests on synthetic data; tests marked `integration` need NiftyReg/NiftySeg
ruff check . && ruff format --check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md). `scripts/` and `examples/` contain the
batch scripts (leave-one-out, SLURM job arrays, NMM atlas preparation) used in the published experiments; they
take their paths as arguments.

## Licence

BSD 3-Clause, see [LICENSE](LICENSE). NiftyReg and NiftySeg have their own licences.
