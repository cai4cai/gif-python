"""``gif-python fetch-mindboggle``: download the Mindboggle-101 volumes and prepare the atlas folder."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

HELP = "Download the Mindboggle-101 volumes from Harvard Dataverse (no login) into the atlas folder layout."

EPILOG = """\
Downloads the five volume archives (about 3.3 GB, 101 subjects) of the Mindboggle-101 data set
(doi:10.7910/DVN/HMQKCK), verifies their SHA-256 against the checksums.sha256 shipped with this package,
extracts t1weighted.nii.gz, t1weighted_brain.nii.gz and labels.DKT31.manual+aseg.nii.gz of every
subject into DATA_DIR/atlases/Mindboggle101/<subject>/ and copies structures_info.csv, tissues_info.csv and
an ITK-SNAP label description file next to them. The data are not hosted by this project; Mindboggle-101
has its own licence and citation requirements (see https://mindboggle.info).

With --clean the sparse labels are replaced afterwards (see 'gif-python clean-labels') and the cleaned
labels are checked against the shipped checksums; their checksums are recorded in cleaned_labels.sha256.

Examples:
  gif-python fetch-mindboggle --clean
  gif-python fetch-mindboggle --only OASIS-TRT-20 --data-dir /scratch/data
  gif-python fetch-mindboggle --list
"""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from src.datasets.mindboggle import ARCHIVES

    parser.epilog = EPILOG
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="default: ./data")
    parser.add_argument("--atlas-dir", type=Path, default=None, help="default: DATA_DIR/atlases/Mindboggle101")
    parser.add_argument(
        "--download-dir", type=Path, default=None, help="where the archives are kept (default: DATA_DIR/downloads/mindboggle101)"
    )
    parser.add_argument("--only", nargs="+", choices=sorted(ARCHIVES), metavar="NAME", help="fetch only these archives")
    parser.add_argument("--all-files", action="store_true", help="extract every file of the archives, not only the three used")
    parser.add_argument("--clean", action="store_true", help="also create labels.DKT31.manual+aseg_cleaned.nii.gz")
    parser.add_argument("--labels", nargs="+", type=int, default=None, help="labels removed by --clean (default: the six sparse labels)")
    parser.add_argument("--skip-verify", action="store_true", help="do not verify the checksums (not recommended)")
    parser.add_argument("--delete-archives", action="store_true", help="delete each archive after extraction")
    parser.add_argument("--list", action="store_true", help="list the archives and exit")
    parser.add_argument(
        "--write-checksums",
        type=Path,
        default=None,
        metavar="FILE",
        help="(maintainers) write the computed SHA-256 of the fetched archives and cleaned labels to FILE",
    )


def run(args: argparse.Namespace) -> int:
    from src.datasets.mindboggle import (
        ARCHIVES,
        DEFAULT_FILES,
        fetch_mindboggle,
        format_checksums,
        load_shipped_checksums,
    )

    if args.list:
        shipped = load_shipped_checksums()
        for archive in ARCHIVES.values():
            digest = shipped.get(archive.filename, "(not recorded)")
            print(f"{archive.name:14s} {archive.n_subjects:3d} subjects {archive.size / 1e6:7.0f} MB  {archive.url}\n{'':14s} sha256 {digest}")
        return 0

    atlas_dir = args.atlas_dir or args.data_dir / "atlases" / "Mindboggle101"
    download_dir = args.download_dir or args.data_dir / "downloads" / "mindboggle101"
    result = fetch_mindboggle(
        atlas_dir,
        download_dir,
        archives=args.only,
        files=None if args.all_files else DEFAULT_FILES,
        clean=args.clean,
        clean_labels=args.labels,
        verify=not args.skip_verify,
        delete_archives=args.delete_archives,
    )
    if args.write_checksums:
        merged = dict(result.archive_checksums)
        merged.update(result.cleaned_checksums)
        args.write_checksums.write_text(format_checksums(merged))
        print(f"wrote {args.write_checksums}")
    logging.getLogger(__name__).info("%d subjects are in %s", len(result.subjects), atlas_dir)
    print(f"Mindboggle-101: {len(result.subjects)} subjects in {atlas_dir}")
    print(f"Next: gif-python segment --atlas-dir {atlas_dir} --img <image> --out <folder>")
    return 0
