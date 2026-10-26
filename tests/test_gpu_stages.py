"""Stage tests of the CUDA program against gif-python's own Python/numba functions (need a CUDA device).

The registration is bypassed with ``--cpp-dir``: every atlas gets the identity control point grid, so the warped atlas images
and label maps are the atlas images and label maps and the Python functions of the CPU backend can be applied to the same data:
``get_lncc_distance`` (weights), ``get_multi_atlas_proba_seg`` / ``get_multi_atlas_tissue_prior`` (fusion, tissue prior) and
``get_structure_seg_from_tissue_seg`` (final parcellation).
"""

import subprocess

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from src.multi_atlas.gpu_backend import find_gpu_binary, read_tables, write_plan
from src.multi_atlas.multi_atlas_fusion import (
    get_multi_atlas_proba_seg,
    get_multi_atlas_tissue_prior,
    get_structure_seg_from_tissue_seg,
)
from src.multi_atlas.utils import get_lncc_distance
from src.utils.config import GifConfig

pytestmark = pytest.mark.gpu

SHAPE = (44, 38, 32)
AFFINE = np.array([[1.1, 0, 0, -20.0], [0, 1.0, 0, 5.0], [0, 0, 1.4, 3.0], [0, 0, 0, 1]])
LABELS = [0, 2, 3, 7, 11, 12]
STRUCTURES = 'label,name,tissues\n0,bg,[0]\n2,A,[3]\n3,B,[2]\n7,C,"[1, 2]"\n11,D,[4]\n12,E,"[3, 4]"\n'
TISSUES = "label,name\n0,Other\n1,Csf\n2,Gm\n3,Wm\n4,Deep\n"


def smooth_field(rng, sigma):
    f = gaussian_filter(rng.normal(size=SHAPE).astype(np.float32), sigma)
    return (f - f.mean()) / f.std()


def identity_grid(path, spacing_voxels=5):
    """Control point grid of reg_f3d (spacing in voxels of the target) that represents the identity transformation."""
    zooms = np.sqrt((AFFINE[:3, :3] ** 2).sum(0))
    n = np.ceil(np.array(SHAPE) * zooms / (spacing_voxels * zooms)).astype(int) + 3
    s = AFFINE.copy()
    s[:3, :3] = AFFINE[:3, :3] * spacing_voxels
    s[:3, 3] = AFFINE[:3, 3] - s[:3, :3] @ np.ones(3)
    ijk = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), axis=-1).astype(np.float64)
    world = ijk @ s[:3, :3].T + s[:3, 3]
    data = np.zeros((*n, 1, 3), np.float32)
    data[..., 0, :] = world
    img = nib.Nifti1Image(data, s)
    img.header["pixdim"][1:4] = zooms * spacing_voxels
    nib.save(img, str(path))


@pytest.fixture(scope="module")
def stage_case(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stages")
    rng = np.random.default_rng(7)
    base = 100 + 40 * smooth_field(rng, 3)
    target = (base + 6 * rng.normal(size=SHAPE)).astype(np.float32)
    (tmp / "structures_info.csv").write_text(STRUCTURES)
    (tmp / "tissues_info.csv").write_text(TISSUES)
    atlases = []
    for k in range(5):
        img = np.rint(base + 12 * smooth_field(rng, 2) + 6 * rng.normal(size=SHAPE)).astype(np.int16)
        field = smooth_field(rng, 2.5)
        lab = np.take(
            np.array(LABELS, np.float32), np.clip(((field + 1.8) / 3.6 * len(LABELS)).astype(int), 0, len(LABELS) - 1)
        )
        lab[:3] = 0  # some background voxels in every atlas
        d = tmp / f"atl{k}"
        d.mkdir()
        nib.save(nib.Nifti1Image(img, AFFINE), str(d / "img.nii.gz"))
        nib.save(nib.Nifti1Image(lab.astype(np.float32), AFFINE), str(d / "seg.nii.gz"))
        identity_grid(d / "cpp.nii.gz")
        atlases.append(
            {
                "name": f"atl{k}",
                "img_path": str(d / "img.nii.gz"),
                "seg_path": str(d / "seg.nii.gz"),
                "img": img,
                "lab": lab,
            }
        )
    nib.save(nib.Nifti1Image(target, AFFINE), str(tmp / "target.nii.gz"))
    structure_df, num_tissue = read_tables(tmp / "structures_info.csv", tmp / "tissues_info.csv")
    write_plan(tmp / "plan.txt", structure_df, num_tissue)
    out = tmp / "out"
    cmd = [
        find_gpu_binary(GifConfig()),
        "--img",
        str(tmp / "target.nii.gz"),
        "--out",
        str(out),
        "--plan",
        str(tmp / "plan.txt"),
    ]
    cmd += ["--cpp-dir", str(tmp), "--dump-dir", str(tmp / "dump")]
    for a in atlases:
        cmd += ["--atlas", a["name"], a["img_path"], a["seg_path"]]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return {"tmp": tmp, "out": out, "target": target, "atlases": atlases, "structure_df": structure_df}


def load(path):
    return np.asarray(nib.load(str(path)).dataobj)


def test_resampling_with_the_identity_grid_reproduces_the_atlas(stage_case):
    for a in stage_case["atlases"]:
        warped = load(stage_case["tmp"] / "dump" / a["name"] / "warped_atlas_img_linear_interp.nii.gz")
        assert (
            np.abs(warped - a["img"]).max() <= 1.0
        )  # linear interpolation at (nearly) integer positions, rounded like int16
        assert (warped == a["img"]).mean() > 0.999
        seg = load(stage_case["out"] / "registered_atlases" / a["name"] / "warped_atlas_seg.nii.gz")
        assert (seg == a["lab"]).mean() > 0.999


def test_lncc_weights_match_get_lncc_distance(stage_case):
    config = GifConfig()
    zooms = nib.load(str(stage_case["tmp"] / "target.nii.gz")).header.get_zooms()[:3]
    for a in stage_case["atlases"]:
        gpu = load(stage_case["tmp"] / "dump" / a["name"] / "weights.nii.gz")
        distance = get_lncc_distance(stage_case["target"], None, a["img"].astype(np.float32), zooms, config)
        ref = np.exp(-distance / config.weights_temperature)
        ref[np.isnan(ref)] = 0
        np.testing.assert_allclose(gpu, ref, rtol=2e-4, atol=1e-6)


def _reference(stage_case):
    config = GifConfig()
    zooms = nib.load(str(stage_case["tmp"] / "target.nii.gz")).header.get_zooms()[:3]
    weights = []
    for a in stage_case["atlases"]:
        distance = get_lncc_distance(stage_case["target"], None, a["img"].astype(np.float32), zooms, config)
        w = np.exp(-distance / config.weights_temperature)
        w[np.isnan(w)] = 0
        weights.append(w)
    warped = np.array([a["lab"].astype(np.uint16) for a in stage_case["atlases"]])
    labels_array = np.array(sorted(np.unique(warped)), dtype=np.uint16)
    proba = get_multi_atlas_proba_seg(
        warped, np.array(weights), labels_array, np.zeros((*SHAPE, len(labels_array)), np.float32)
    )
    structure_dict = stage_case["structure_df"].to_dict(orient="dict")
    prior = get_multi_atlas_tissue_prior(proba, labels_array, structure_dict, 5, None)
    return labels_array, proba, structure_dict, prior


def test_fusion_and_tissue_prior_match_the_numba_functions(stage_case):
    labels_array, proba, _, prior = _reference(stage_case)
    argmax = labels_array[np.argmax(proba, axis=-1)]
    gpu_argmax = load(stage_case["out"] / "multi_atlas_proba_seg_argmax.nii.gz")
    assert (gpu_argmax != argmax).mean() < 1e-3
    gpu_prior = load(stage_case["out"] / "multi_atlas_tissue_prior.nii.gz")
    assert gpu_prior.shape == prior.shape
    assert (np.abs(gpu_prior - prior).max(-1) > 1e-3).mean() < 1e-3
    np.testing.assert_allclose(gpu_prior.sum(-1), 1.0, atol=1e-5)


def test_final_parcellation_matches_get_structure_seg_from_tissue_seg(stage_case):
    labels_array, proba, structure_dict, prior = _reference(stage_case)
    gpu_prior = load(stage_case["out"] / "multi_atlas_tissue_prior.nii.gz")
    post = load(stage_case["out"] / "multi_atlas_tissue_seg.nii.gz").astype(np.float16)
    post[gpu_prior[..., 0] == 1] = 1
    post[[0, -1], :, :] = 1
    post[:, [0, -1], :] = 1
    post[:, :, [0, -1]] = 1
    tseg = np.argmax(post, axis=-1).astype(np.uint16)
    expected = get_structure_seg_from_tissue_seg(tseg, proba, labels_array, structure_dict["tissues"])
    gpu = load(stage_case["out"] / "final_parcellation.nii.gz")
    assert gpu.dtype == np.int32
    # labels of zero probability are chosen in a different (arbitrary) order by numpy's argpartition: allow a small fraction
    assert (gpu != expected).mean() < 0.02
