"""
Run GIF multi-atlas segmentation for Mindboggle42 using Mindboggle59 atlases.
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Iterable

from src.multi_atlas.multi_atlas_segmentation import multi_atlas_segmentation


ATLAS_ROOT = "/nfs/home/jwang/datasets/Mindboggle101/subjects"
RESULTS_ROOT = "outputs/mb59_to_mb42_mindboggle101_gif"

IMAGE_NAME = "t1weighted_brain.nii.gz"
LABEL_NAME = "labels.DKT31.manual+aseg_cleaned.nii.gz"
EXPECTED_ATLAS_COUNT = 59
EXPECTED_TARGET_COUNT = 42

MB42_SUBJECTS = [
    "NKI-RS-22-1",
    "NKI-RS-22-2",
    "NKI-RS-22-3",
    "NKI-RS-22-4",
    "NKI-RS-22-5",
    "NKI-RS-22-6",
    "NKI-RS-22-7",
    "NKI-RS-22-8",
    "NKI-RS-22-9",
    "NKI-RS-22-10",
    "NKI-RS-22-11",
    "NKI-RS-22-12",
    "NKI-RS-22-13",
    "NKI-RS-22-14",
    "NKI-RS-22-15",
    "NKI-RS-22-16",
    "NKI-RS-22-17",
    "NKI-RS-22-18",
    "NKI-RS-22-19",
    "NKI-RS-22-20",
    "NKI-RS-22-21",
    "NKI-RS-22-22",
    "NKI-TRT-20-1",
    "NKI-TRT-20-2",
    "NKI-TRT-20-3",
    "NKI-TRT-20-4",
    "NKI-TRT-20-5",
    "NKI-TRT-20-6",
    "NKI-TRT-20-7",
    "NKI-TRT-20-8",
    "NKI-TRT-20-9",
    "NKI-TRT-20-10",
    "NKI-TRT-20-11",
    "NKI-TRT-20-12",
    "NKI-TRT-20-13",
    "NKI-TRT-20-14",
    "NKI-TRT-20-15",
    "NKI-TRT-20-16",
    "NKI-TRT-20-17",
    "NKI-TRT-20-18",
    "NKI-TRT-20-19",
    "NKI-TRT-20-20",
]

MB59_SUBJECTS = [
    "Afterthought-1",
    "Colin27-1",
    "HLN-12-1",
    "HLN-12-2",
    "HLN-12-3",
    "HLN-12-4",
    "HLN-12-5",
    "HLN-12-6",
    "HLN-12-7",
    "HLN-12-8",
    "HLN-12-9",
    "HLN-12-10",
    "HLN-12-11",
    "HLN-12-12",
    "MMRR-21-1",
    "MMRR-21-2",
    "MMRR-21-3",
    "MMRR-21-4",
    "MMRR-21-5",
    "MMRR-21-6",
    "MMRR-21-7",
    "MMRR-21-8",
    "MMRR-21-9",
    "MMRR-21-10",
    "MMRR-21-11",
    "MMRR-21-12",
    "MMRR-21-13",
    "MMRR-21-14",
    "MMRR-21-15",
    "MMRR-21-16",
    "MMRR-21-17",
    "MMRR-21-18",
    "MMRR-21-19",
    "MMRR-21-20",
    "MMRR-21-21",
    "MMRR-3T7T-2-1",
    "MMRR-3T7T-2-2",
    "OASIS-TRT-20-1",
    "OASIS-TRT-20-2",
    "OASIS-TRT-20-3",
    "OASIS-TRT-20-4",
    "OASIS-TRT-20-5",
    "OASIS-TRT-20-6",
    "OASIS-TRT-20-7",
    "OASIS-TRT-20-8",
    "OASIS-TRT-20-9",
    "OASIS-TRT-20-10",
    "OASIS-TRT-20-11",
    "OASIS-TRT-20-12",
    "OASIS-TRT-20-13",
    "OASIS-TRT-20-14",
    "OASIS-TRT-20-15",
    "OASIS-TRT-20-16",
    "OASIS-TRT-20-17",
    "OASIS-TRT-20-18",
    "OASIS-TRT-20-19",
    "OASIS-TRT-20-20",
    "Twins-2-1",
    "Twins-2-2",
]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Mindboggle42 parcellation with the Mindboggle59 atlas set."
    )
    parser.add_argument("--atlas-root", default=ATLAS_ROOT)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def validate_subject_files(atlas_root: str, subjects: list[str]) -> None:
    missing = []
    for subject in subjects:
        subject_dir = os.path.join(atlas_root, subject)
        image_path = os.path.join(subject_dir, IMAGE_NAME)
        label_path = os.path.join(subject_dir, LABEL_NAME)
        if not os.path.isfile(image_path) or not os.path.isfile(label_path):
            missing.append(subject)
    if missing:
        raise FileNotFoundError(
            "Missing image or cleaned label for: " + ", ".join(missing)
        )


def build_atlas_dicts(atlas_root: str) -> list[dict[str, str]]:
    validate_subject_files(atlas_root, MB59_SUBJECTS)
    return [
        {
            "name": subject,
            "img_path": os.path.join(atlas_root, subject, IMAGE_NAME),
            "seg_path": os.path.join(atlas_root, subject, LABEL_NAME),
        }
        for subject in MB59_SUBJECTS
    ]


def run_mb42_segmentation(
    atlas_root: str = ATLAS_ROOT,
    results_root: str = RESULTS_ROOT,
    limit: int | None = None,
    skip_existing: bool = False,
) -> None:
    if len(MB59_SUBJECTS) != EXPECTED_ATLAS_COUNT:
        raise ValueError(f"Expected {EXPECTED_ATLAS_COUNT} MB59 atlases")
    if len(MB42_SUBJECTS) != EXPECTED_TARGET_COUNT:
        raise ValueError(f"Expected {EXPECTED_TARGET_COUNT} MB42 targets")

    validate_subject_files(atlas_root, MB42_SUBJECTS)
    structure_info_csv = os.path.join(atlas_root, "structures_info.csv")
    tissue_info_csv = os.path.join(atlas_root, "tissues_info.csv")

    targets = MB42_SUBJECTS if limit is None else MB42_SUBJECTS[:limit]
    os.makedirs(results_root, exist_ok=True)

    for subject in targets:
        img_path = os.path.join(atlas_root, subject, IMAGE_NAME)
        results_dir = os.path.join(results_root, subject)
        parcellation_file = os.path.join(results_dir, "final_parcellation.nii.gz")
        if skip_existing and os.path.isfile(parcellation_file):
            print(f"{subject}: already processed - skipping")
            continue

        os.makedirs(results_dir, exist_ok=True)
        atlas_paths_dicts_list = build_atlas_dicts(atlas_root)
        print(
            f"{subject}: running segmentation with "
            f"{len(atlas_paths_dicts_list)} MB59 atlases"
        )
        multi_atlas_segmentation(
            img_path=img_path,
            mask_path=None,
            atlas_paths_dicts_list=atlas_paths_dicts_list,
            structure_info_csv_path=structure_info_csv,
            tissue_info_csv_path=tissue_info_csv,
            save_dir=results_dir,
        )
        print(f"{subject}: completed, results saved to {results_dir}")


if __name__ == "__main__":
    args = parse_args()
    run_mb42_segmentation(
        atlas_root=args.atlas_root,
        results_root=args.results_root,
        limit=args.limit,
        skip_existing=args.skip_existing,
    )
