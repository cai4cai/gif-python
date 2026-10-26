"""End to end run on tiny synthetic volumes. Needs NiftyReg and NiftySeg (skipped otherwise)."""

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from src.multi_atlas.multi_atlas_segmentation import multi_atlas_segmentation
from src.utils.config import GifConfig

pytestmark = pytest.mark.integration


def make_head(shift=(0, 0, 0), shape=(48, 48, 48), seed=0):
    """Three nested blobs (CSF-ish shell, grey matter, white matter core) with some texture."""
    rng = np.random.default_rng(seed)
    grid = np.indices(shape).astype(np.float32)
    centre = (
        np.array(shape, dtype=np.float32)[:, None, None, None] / 2
        + np.array(shift, dtype=np.float32)[:, None, None, None]
    )
    radius = np.sqrt(((grid - centre) ** 2).sum(axis=0))
    labels = np.zeros(shape, dtype=np.uint8)
    labels[radius < 17] = 1  # "cortex"
    labels[radius < 11] = 2  # "white matter"
    intensity = np.zeros(shape, dtype=np.float32)
    intensity[labels == 1] = 100
    intensity[labels == 2] = 200
    intensity = gaussian_filter(intensity, 1.0) + gaussian_filter(rng.normal(0, 3, shape), 1.0)
    return intensity.astype(np.float32), labels


def test_segmentation_of_a_blob_phantom(tmp_path, write_nifti):
    target, target_labels = make_head(seed=1)
    atlases = []
    for i, shift in enumerate([(0, 0, 0), (1, 0, 0), (0, 1, 0)]):
        img, lab = make_head(shift=shift, seed=10 + i)
        atlases.append(
            {
                "name": f"atlas{i}",
                "img_path": write_nifti(img, tmp_path / f"atlas{i}.nii.gz"),
                "seg_path": write_nifti(lab, tmp_path / f"atlas{i}_seg.nii.gz"),
            }
        )
    img_path = write_nifti(target, tmp_path / "target.nii.gz")
    (tmp_path / "structures_info.csv").write_text("label,name,tissues\n0,Background,[0]\n1,Cortex,[1]\n2,White,[2]\n")
    (tmp_path / "tissues_info.csv").write_text("label,name\n0,Other\n1,Grey Matter\n2,White Matter\n")

    result = multi_atlas_segmentation(
        img_path=img_path,
        mask_path=None,
        atlas_paths_dicts_list=atlases,
        structure_info_csv_path=tmp_path / "structures_info.csv",
        tissue_info_csv_path=tmp_path / "tissues_info.csv",
        save_dir=tmp_path / "out",
        config=GifConfig(multiprocessing=False, omp=1, reg_f3d_maxit=50, reg_f3d_ln=2, reg_f3d_lp=2),
    )
    saved = nib.load(str(tmp_path / "out" / "final_parcellation.nii.gz")).get_fdata()
    np.testing.assert_array_equal(saved, result)
    assert result.shape == target.shape
    inside = target_labels > 0
    agreement = (result[inside] == target_labels[inside]).mean()
    assert agreement > 0.7
