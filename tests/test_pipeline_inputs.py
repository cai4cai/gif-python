"""Input validation and structure/tissue information handling of multi_atlas_segmentation.

These checks run before any NiftyReg tool is started, so they work without the binaries.
"""

import numpy as np
import pytest

from src.multi_atlas.multi_atlas_segmentation import multi_atlas_segmentation
from src.utils.config import GifConfig

STRUCTURES = "label,name,tissues\n0,Background,[0]\n5,Left-A,[1]\n7,Left-B,\"[1, 2]\"\n"
TISSUES = "label,name\n0,Other\n1,Grey Matter\n2,White Matter\n"


@pytest.fixture
def case(tmp_path, write_nifti):
    """A target image and two atlases with labels 0, 5, 7, plus matching csv files."""
    shape = (8, 8, 8)
    labels = np.zeros(shape, dtype=np.uint8)
    labels[2:4] = 5
    labels[4:6] = 7
    img = write_nifti(np.random.default_rng(0).random(shape).astype(np.float32), tmp_path / "img.nii.gz")
    atlases = []
    for name in ("a1", "a2"):
        atlases.append(
            {
                "name": name,
                "img_path": write_nifti(np.ones(shape, np.float32), tmp_path / f"{name}.nii.gz"),
                "seg_path": write_nifti(labels, tmp_path / f"{name}_seg.nii.gz"),
            }
        )
    structures = tmp_path / "structures_info.csv"
    structures.write_text(STRUCTURES)
    tissues = tmp_path / "tissues_info.csv"
    tissues.write_text(TISSUES)
    return {
        "img_path": img,
        "mask_path": None,
        "atlas_paths_dicts_list": atlases,
        "structure_info_csv_path": structures,
        "tissue_info_csv_path": tissues,
        "save_dir": tmp_path / "out",
        "config": GifConfig(multiprocessing=False, niftyreg_path=str(tmp_path / "no_such_dir")),
    }


def test_missing_target_image(case):
    case["img_path"] = "does_not_exist.nii.gz"
    with pytest.raises(FileNotFoundError, match="Input image"):
        multi_atlas_segmentation(**case)


def test_missing_mask(case):
    case["mask_path"] = "does_not_exist.nii.gz"
    with pytest.raises(FileNotFoundError, match="Input mask"):
        multi_atlas_segmentation(**case)


def test_missing_atlas_segmentation(case):
    case["atlas_paths_dicts_list"][1]["seg_path"] = "nope.nii.gz"
    with pytest.raises(FileNotFoundError, match="Atlas segmentation"):
        multi_atlas_segmentation(**case)


def test_missing_atlas_mask(case):
    case["atlas_paths_dicts_list"][0]["mask_path"] = "nope.nii.gz"
    with pytest.raises(FileNotFoundError, match="Atlas mask"):
        multi_atlas_segmentation(**case)


def test_missing_csv_files(case, tmp_path):
    case["structure_info_csv_path"] = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="Structure info"):
        multi_atlas_segmentation(**case)


def test_label_in_atlas_but_not_in_structure_info(case, write_nifti, tmp_path):
    labels = np.zeros((8, 8, 8), dtype=np.uint8)
    labels[0:2] = 9  # not in structures_info.csv
    case["atlas_paths_dicts_list"][0]["seg_path"] = write_nifti(labels, tmp_path / "bad_seg.nii.gz")
    with pytest.raises(ValueError, match=r"not in the structure info csv file.*\[9\]"):
        multi_atlas_segmentation(**case)


def test_tissue_in_structure_info_but_not_in_tissue_info(case):
    case["tissue_info_csv_path"].write_text("label,name\n0,Other\n1,Grey Matter\n")
    with pytest.raises(ValueError, match=r"not in the tissue info csv file.*\[2\]"):
        multi_atlas_segmentation(**case)


def test_caller_dictionaries_are_not_modified(case):
    atlases = case["atlas_paths_dicts_list"]
    with pytest.raises(Exception):  # stops at the registration (no such NiftyReg directory)
        multi_atlas_segmentation(**case)
    assert all("mask_path" not in atlas for atlas in atlases)


def test_missing_niftyreg_gives_clear_error(case):
    with pytest.raises(FileNotFoundError, match="reg_aladin"):
        multi_atlas_segmentation(**case)
