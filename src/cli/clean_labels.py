"""``gif-python clean-labels``: remove sparse labels from segmentations."""

from __future__ import annotations

import argparse
import logging

HELP = "Replace sparse labels by the most frequent neighbouring label (Mindboggle-101 clean-up)."

EPILOG = """\
Every voxel with one of the labels to remove gets the most frequent label among its 26 neighbours
(neighbours with the same label are ignored; ties go to the smallest label). The default labels are
the six sparse cortical labels 1001, 1032, 1033, 2001, 2032 and 2033. The labels 72 (5th ventricle),
80 (non-WM hypointensities) and 77 (WM hypointensities) are only removed if you ask for them with
--labels.

Example:
  gif-python clean-labels "data/atlases/Mindboggle101/*/labels.DKT31.manual+aseg.nii.gz"
  gif-python clean-labels "data/atlases/Mindboggle101/*/labels.DKT31.manual+aseg.nii.gz" \\
      --labels 1001 1032 1033 2001 2032 2033 72 80 77 --suffix _cleaned_all
"""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.epilog = EPILOG
    parser.add_argument("inputs", nargs="+", metavar="INPUT", help="NIfTI files or glob patterns (quote the patterns)")
    parser.add_argument(
        "--labels",
        nargs="+",
        type=int,
        default=None,
        metavar="LABEL",
        help="labels to remove (default: 1001 1032 1033 2001 2032 2033)",
    )
    parser.add_argument(
        "--suffix", default="_cleaned", help="appended to the file name stem of the output (default: %(default)s)"
    )
    parser.add_argument("--dry-run", action="store_true", help="report the number of voxels to change, write nothing")


def run(args: argparse.Namespace) -> int:
    from src.labels.cleaning import DEFAULT_SPARSE_LABELS, clean_labels_file, expand_inputs, output_path

    labels = args.labels if args.labels else list(DEFAULT_SPARSE_LABELS)
    files = expand_inputs(args.inputs, args.suffix)
    logging.getLogger(__name__).info("Removing labels %s from %d file(s)", labels, len(files))
    total = 0
    for path in files:
        if args.dry_run:
            import nibabel as nib
            import numpy as np

            from src.labels.cleaning import clean_labels_array

            _, n_changed = clean_labels_array(np.asanyarray(nib.load(str(path)).dataobj), labels)
            out = output_path(path, args.suffix)
            print(f"{path}: would change {n_changed} voxels -> {out}")
        else:
            out, n_changed = clean_labels_file(path, labels_to_replace=labels, suffix=args.suffix)
            print(f"{path}: changed {n_changed} voxels -> {out}")
        total += n_changed
    print(f"Total changed voxels: {total}")
    return 0
