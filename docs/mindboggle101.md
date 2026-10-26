# Mindboggle-101

How to use the Mindboggle-101 manually labelled brains (Klein and Tourville, 2012;
[mindboggle.info](https://mindboggle.info)) as atlases.

## Quickstart

```bash
pip install .                         # see INSTALL.md for NiftyReg and NiftySeg
gif-python fetch-mindboggle --clean   # about 3.3 GB; verified against checksums.sha256
```

This creates (data are read from Harvard Dataverse, doi:10.7910/DVN/HMQKCK; nothing is hosted by this project):

```
data/atlases/Mindboggle101/
    structures_info.csv  tissues_info.csv  itksnap_label_descriptions_mindboggle101.txt
    OASIS-TRT-20-1/  t1weighted.nii.gz  t1weighted_brain.nii.gz
                     labels.DKT31.manual+aseg.nii.gz  labels.DKT31.manual+aseg_cleaned.nii.gz
    ... (101 subjects)
data/downloads/mindboggle101/*.tar.gz           (the verified archives)
```

Segment one subject with the other 100 as atlases (leave-one-out):

```bash
gif-python segment \
    --img data/atlases/Mindboggle101/OASIS-TRT-20-1/t1weighted_brain.nii.gz \
    --atlas-dir data/atlases/Mindboggle101 --exclude-atlas OASIS-TRT-20-1 \
    --out outputs/OASIS-TRT-20-1
```

The defaults use `t1weighted_brain.nii.gz` and the cleaned labels `labels.DKT31.manual+aseg_cleaned.nii.gz` of the
atlases and `structures_info.csv` / `tissues_info.csv` from the atlas folder; the target should be prepared like
the atlases (here: brain extracted). A run with 100 atlases is long (registration of every atlas); try
`--exclude-atlas` several times to use a few atlases first, and see [tuning-registration.md](tuning-registration.md).
Results: `outputs/OASIS-TRT-20-1/final_parcellation.nii.gz` (labels of the atlases) and
`multi_atlas_tissue_seg.nii.gz`. View them with ITK-SNAP and `itksnap_label_descriptions_mindboggle101.txt`
(`File > Import > Label Descriptions`), or add `--view`.

Options of `fetch-mindboggle`: `--only OASIS-TRT-20 ...` (subset of the five archives), `--data-dir`,
`--atlas-dir`, `--download-dir`, `--all-files`, `--delete-archives`, `--list`. The archive checksums are
checked before anything is extracted; a mismatch stops the run. Without `--clean` only the original labels are prepared.

## Label tables

`structures_info.csv` (label, name, tissues) and `tissues_info.csv` (tissue numbers) are in
`src/resources/mindboggle101/`, with the provenance and the tissue assignment of every group of structures
documented in the README of that folder. The table lists all 114 labels of the 101 volumes (the labels
checked against the downloaded volumes), including the six sparse cortical labels 1001, 1032, 1033, 2001, 2032 and 2033
(grey matter, as agreed in cai4cai/gif-python#1) and the problematic 72, 77 and 80. Labels of the table that are not in the atlases used
in a run are only reported.

For another atlas, generate the tables from its segmentations and a FreeSurfer colour table:

```bash
gif-python make-structure-info data/atlases/MyAtlas/*/parcellation.nii.gz \
    --color-table FreeSurferColorLUT.txt --out-dir data/atlases/MyAtlas
```

The tissue classes are assigned from the structure names with built-in rules; labels that match no rule are
listed in a warning and given all tissues, so check the output (`--strict` makes this an error; `--tissue-rules` adds your rules).

## Cleaning the sparse labels

Six labels (1001, 1032, 1033, 2001, 2032, 2033) exist in only a handful of subjects (in the 101 volumes: 77 voxels
in 28 subjects). `clean-labels` replaces each voxel of these labels by the most frequent label among its 26
neighbours (ties: the smallest label):

```bash
gif-python clean-labels "data/atlases/Mindboggle101/*/labels.DKT31.manual+aseg.nii.gz"
```

The result is identical to the per-voxel loop this replaces (`clean_small_labels.py`; checked on all 28 affected
subjects and in the tests), and `fetch-mindboggle --clean` checks the cleaned volumes against the shipped checksums.
The output name is `<input>_cleaned.nii.gz` (`--suffix` changes it), `--labels` chooses other labels, `--dry-run`
only counts.

The labels 72 (5th ventricle), 80 (non-WM hypointensities) and 77 (WM hypointensities) are **not** removed by default.
They can be requested with `--labels`, but the neighbour rule is not a good fit for them: on the 101 volumes it
stops with an error ("no neighbour with a different label", or a most frequent neighbour that is itself in the list) for
1 volume with `--labels 72`, 1 volume with `--labels 80`, and 58 volumes with `--labels 77`, which is a large structure
(166,685 voxels in the 101 subjects, against 584 for label 72 and 3,024 for label 80). Whether and how these labels
should be removed or replaced is left to the owners of the data set.

## Checksums

`src/resources/mindboggle101/checksums.sha256` has two parts, explained in its header: the SHA-256 of the five archives
(`sha256sum -c --ignore-missing checksums.sha256` in the folder with the archives) and, for the cleaned labels
of every subject, a SHA-256 of the data type, shape and voxel values, which does not depend on the file format version.
`fetch-mindboggle --clean` writes the values it computed to `cleaned_labels.sha256` in the atlas folder.
