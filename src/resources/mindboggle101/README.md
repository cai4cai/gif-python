# Mindboggle-101 label tables

Files for atlases built from the Mindboggle-101 volumes (`labels.DKT31.manual+aseg.nii.gz`):

| File | Purpose |
|---|---|
| `structures_info.csv` | one row per label: `label,name,tissues`; the tissues are the tissue classes the structure may belong to |
| `tissues_info.csv` | the six tissue classes (0 non-brain outer tissue, 1 CSF, 2 grey matter, 3 white matter, 4 deep grey matter, 5 brain stem and pons) |
| `itksnap_label_descriptions_mindboggle101.txt` | ITK-SNAP label description file (`File > Import > Label Descriptions`) |
| `checksums.sha256` | SHA-256 of the five Harvard Dataverse volume archives and of the cleaned labels of all 101 subjects (see its header for exactly what is hashed) |

`gif-python fetch-mindboggle` copies the two csv files and the label description file into the atlas
folder; they can also be used directly from the installed package
(`from src.resources import mindboggle101_file`).

## Provenance and choices

* `structures_info.csv` is the table posted in cai4cai/gif-python#1 (`mindboggle_atlases_structures_info.csv`)
  plus the six labels that issue reports as missing: 1001 `ctx-lh-bankssts`, 1032 `ctx-lh-frontalpole`,
  1033 `ctx-lh-temporalpole`, 2001 `ctx-rh-bankssts`, 2032 `ctx-rh-frontalpole`, 2033 `ctx-rh-temporalpole`.
  Their names follow `FreeSurferColorLUT.txt`; since they are cortical (`ctx-`) they are assigned to
  grey matter, `[2]`, as agreed in the issue. Rows that were already in the posted table are unchanged.
* Tissue assignments of the other structures are those of the posted table. In particular:
  cortex `[2]`; cerebral and cerebellar white matter, corpus callosum segments and
  WM-hypointensities (77) `[3]`; ventricles, CSF and the 5th ventricle (72) `[1]`;
  thalamus, caudate, putamen, pallidum and accumbens `[3, 4]`; hippocampus and amygdala `[2, 4]`;
  ventral DC `[3, 4, 5]`; brain stem `[5]`; vessels and the optic chiasm (85) `[4]`;
  choroid plexus `[1, 2, 4, 5]`; non-WM-hypointensities (80) `[2, 4, 5]`;
  `ctx-lh-unknown` and `ctx-rh-unknown` (1000, 2000) may belong to any tissue.
* All labels of the volumes are listed, including the sparse ones that `clean-labels` removes and
  the problematic 72, 77 and 80. A label that is in the table but not in a given set of atlases only
  produces an informational message, so the same table works with the original and the cleaned labels.
* The checksums were computed from complete downloads of the archives on 2026-10-26 (their MD5 sums equal those
  published by Dataverse). The cleaned-label checksums were produced by `clean-labels` with the default labels.
  Labels present in the 101 volumes are exactly the 114 labels of `structures_info.csv`.
* The ITK-SNAP file is the one posted in the same issue (its colours are kept); the four labels it
  lacked (1001, 1032, 1033, 2001) were added with otherwise unused colours. It lists exactly the labels
  of `structures_info.csv`, with the same names.

These are conventions, not ground truth: they are the choices of the Mindboggle owners of this
repository, documented here so they can be reviewed. The mapping of structures to tissues has an
effect on the EM refinement of the final segmentation.
