"""
Basic script to run multi-atlas segmentation on a single subject.
"""
import os
from glob import glob

from src.multi_atlas.multi_atlas_segmentation import multi_atlas_segmentation

# ----------------------
# Configuration
# ----------------------
# Input: single subject directory
SUBJECT_DIR = "data/input/subject-01"

# Atlases: Mindboggle atlas directory
ATLAS_ROOT = "data/atlases/Mindboggle101"

# Output: where to save results
RESULTS_ROOT = "results"

# ----------------------
# Run segmentation
# ----------------------
def run_segmentation():
    """Run multi-atlas segmentation on a single subject."""
    subject_name = os.path.basename(SUBJECT_DIR)
    print(f"Processing: {subject_name}")

    # Input file
    img_path = os.path.join(SUBJECT_DIR, "t1weighted_brain.nii.gz")

    # Output directory
    results_dir = os.path.join(RESULTS_ROOT, subject_name)
    os.makedirs(results_dir, exist_ok=True)

    # Check if already processed
    parcellation_file = os.path.join(results_dir, "final_parcellation.nii.gz")
    if os.path.isfile(parcellation_file):
        print("Already processed - skipping")
        return

    # Collect atlas paths
    atlas_dirs = [d for d in glob(os.path.join(ATLAS_ROOT, "*")) if os.path.isdir(d)]
    atlas_paths_dicts_list = [
        {
            'name': os.path.basename(atlas_dir),
            'img_path': os.path.join(atlas_dir, 't1weighted.nii.gz'),
            'seg_path': os.path.join(atlas_dir, 'labels.DKT31.manual+aseg_cleaned.nii.gz')
        }
        for atlas_dir in atlas_dirs
    ]

    # Structure and tissue info CSVs
    structure_info_csv = os.path.join(ATLAS_ROOT, "structures_info.csv")
    tissue_info_csv = os.path.join(ATLAS_ROOT, "tissues_info.csv")

    # Run segmentation
    print(f"Running segmentation with {len(atlas_paths_dicts_list)} atlases...")

    _ = multi_atlas_segmentation(
        img_path=img_path,
        mask_path=None,
        atlas_paths_dicts_list=atlas_paths_dicts_list,
        structure_info_csv_path=structure_info_csv,
        tissue_info_csv_path=tissue_info_csv,
        save_dir=results_dir
    )

    print(f"Completed! Results saved to: {results_dir}")

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    print("="*60)
    print(f"Subject: {SUBJECT_DIR}")
    print(f"Atlases: {ATLAS_ROOT}")
    print(f"Output: {RESULTS_ROOT}")
    print("="*60)

    run_segmentation()

    print("="*60)
    print("Done!")
    print("="*60)
