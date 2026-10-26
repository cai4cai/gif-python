"""Run the multi-atlas segmentation for one image with a folder of atlases.

This replaces the former ``tests/test_multi_atlas_segmentation*.py`` scripts, which differed only in
the folder and file names. Atlas folder layout::

    ATLAS_ROOT/
        structures_info.csv
        tissues_info.csv
        atlas-1/<atlas image>, <atlas segmentation>[, <atlas mask>]
        atlas-2/...

Example (GENFI-style atlases with masks)::

    python examples/run_multi_atlas_segmentation.py --img t1.nii.gz --mask t1_mask.nii.gz \
        --atlas-root data/atlases/GENFI_atlases --atlas-img srr.nii.gz \
        --atlas-seg parcellation.nii.gz --atlas-mask mask.nii.gz --out results/genfi

The ``gif-python segment`` command does the same with more options.
"""

import argparse
import logging
import time
from pathlib import Path

from src.multi_atlas.multi_atlas_segmentation import multi_atlas_segmentation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--img", required=True, type=Path, help="image to segment")
    parser.add_argument("--mask", type=Path, default=None, help="mask of the image to segment (optional)")
    parser.add_argument("--atlas-root", required=True, type=Path, help="folder with one sub-folder per atlas")
    parser.add_argument("--atlas-img", required=True, help="file name of the atlas image in each atlas folder")
    parser.add_argument("--atlas-seg", required=True, help="file name of the atlas segmentation in each atlas folder")
    parser.add_argument(
        "--atlas-mask", default=None, help="file name of the atlas mask in each atlas folder (optional)"
    )
    parser.add_argument("--structure-info", type=Path, default=None, help="default: ATLAS_ROOT/structures_info.csv")
    parser.add_argument("--tissue-info", type=Path, default=None, help="default: ATLAS_ROOT/tissues_info.csv")
    parser.add_argument("--out", required=True, type=Path, help="output folder")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    atlas_dirs = sorted(d for d in args.atlas_root.iterdir() if d.is_dir())
    atlases = []
    for atlas_dir in atlas_dirs:
        atlas = {
            "name": atlas_dir.name,
            "img_path": str(atlas_dir / args.atlas_img),
            "seg_path": str(atlas_dir / args.atlas_seg),
        }
        if args.atlas_mask:
            atlas["mask_path"] = str(atlas_dir / args.atlas_mask)
        atlases.append(atlas)

    time_0 = time.time()
    multi_atlas_segmentation(
        img_path=args.img,
        mask_path=args.mask,
        atlas_paths_dicts_list=atlases,
        structure_info_csv_path=args.structure_info or args.atlas_root / "structures_info.csv",
        tissue_info_csv_path=args.tissue_info or args.atlas_root / "tissues_info.csv",
        save_dir=args.out,
    )
    print(f"Total running time: {time.time() - time_0:.1f} seconds")
    print(f"Multi-atlas segmentation output saved to: {args.out / 'final_parcellation.nii.gz'}")


if __name__ == "__main__":
    main()
