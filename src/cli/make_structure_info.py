"""``gif-python make-structure-info``: create the csv files that describe an atlas."""

from __future__ import annotations

import argparse
import logging

HELP = "Create structures_info.csv and tissues_info.csv from an atlas segmentation and a FreeSurfer colour table."

EPILOG = """\
The labels are read from the segmentation volume(s); names (and ITK-SNAP colours) come from a
FreeSurfer colour table such as FreeSurferColorLUT.txt. The tissue classes of each structure are
assigned from its name with built-in rules that follow the FreeSurfer naming conventions (cortex
-> grey matter, white matter, ventricles -> CSF, deep nuclei -> deep grey matter, ...), the same
rules that reproduce the Mindboggle-101 table. Labels that match no rule are reported and may belong
to all tissues; review them in the output, or use --strict / --tissue-rules.

Example:
  gif-python make-structure-info data/atlases/MyAtlas/*/parcellation.nii.gz \\
      --color-table $FREESURFER_HOME/FreeSurferColorLUT.txt --out-dir data/atlases/MyAtlas
"""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.epilog = EPILOG
    parser.add_argument("segmentations", nargs="+", metavar="SEG", help="atlas segmentation(s) or glob patterns")
    parser.add_argument("--out-dir", required=True, help="folder for structures_info.csv and tissues_info.csv")
    parser.add_argument("--color-table", default=None, help="FreeSurfer colour table (index name R G B A)")
    parser.add_argument(
        "--tissue-rules",
        default=None,
        help="csv with the columns pattern (regular expression on the lower-case name) and tissues, tried first",
    )
    parser.add_argument(
        "--tissues-info", default=None, help="use this tissues_info.csv instead of the default six tissues"
    )
    parser.add_argument("--strict", action="store_true", help="fail if a label has no name or no tissue rule")
    parser.add_argument("--no-itksnap", action="store_true", help="do not write the ITK-SNAP label description file")


def run(args: argparse.Namespace) -> int:
    import glob

    from src.labels.structure_info import make_structure_info

    paths = []
    for pattern in args.segmentations:
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        paths.extend(matches)
    if not paths:
        raise FileNotFoundError(f"No segmentation matches {' '.join(args.segmentations)}")
    written = make_structure_info(
        paths,
        args.out_dir,
        color_table_path=args.color_table,
        tissue_rules_path=args.tissue_rules,
        tissues_info_path=args.tissues_info,
        strict=args.strict,
        write_itksnap=not args.no_itksnap,
    )
    logging.getLogger(__name__).info("Read labels from %d file(s)", len(paths))
    for path in written.values():
        print(f"wrote {path}")
    return 0
