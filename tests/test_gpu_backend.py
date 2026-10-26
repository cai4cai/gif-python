"""GPU backend: Python glue (tests without a GPU use a stand-in executable) and an end-to-end run on a phantom (needs CUDA)."""

import json
import os
import stat
import sys
import textwrap

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from src.cli.main import main
from src.multi_atlas import gpu_backend
from src.multi_atlas.multi_atlas_segmentation import multi_atlas_segmentation
from src.utils.config import BinaryNotFoundError, GifConfig, coerce_field
from src.utils.process import ExternalToolError

STRUCTURES = 'label,name,tissues\n0,Background,[0]\n5,Left-A,[1]\n7,Left-B,"[1, 2]"\n'
TISSUES = "label,name\n0,Other\n1,Grey Matter\n2,White Matter\n"


def make_fake_binary(tmp_path, body):
    """An executable Python script that stands in for gif_gpu."""
    path = tmp_path / "fake_gif_gpu"
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


FAKE_OK = """
    import json, sys
    import nibabel as nib
    import numpy as np
    argv = sys.argv[1:]
    if "--check-device" in argv:
        print("fake device"); sys.exit(0)
    out = argv[argv.index("--out") + 1]
    img = nib.load(argv[argv.index("--img") + 1])
    json.dump(argv, open(out + "/argv.json", "w"))
    plan = open(argv[argv.index("--plan") + 1]).read()
    open(out + "/plan_seen.txt", "w").write(plan)
    nib.save(nib.Nifti1Image(np.full(img.shape, 7, np.int32), img.affine), out + "/final_parcellation.nii.gz")
"""


@pytest.fixture
def case(tmp_path, write_nifti):
    shape = (8, 8, 8)
    labels = np.zeros(shape, dtype=np.uint8)
    labels[2:4] = 5
    labels[4:6] = 7
    img = write_nifti(np.random.default_rng(0).random(shape).astype(np.float32), tmp_path / "img.nii.gz")
    atlases = [
        {
            "name": name,
            "img_path": write_nifti(np.ones(shape, np.float32), tmp_path / f"{name}.nii.gz"),
            "seg_path": write_nifti(labels, tmp_path / f"{name}_seg.nii.gz"),
        }
        for name in ("a1", "a2")
    ]
    (tmp_path / "structures_info.csv").write_text(STRUCTURES)
    (tmp_path / "tissues_info.csv").write_text(TISSUES)
    return {
        "img_path": img,
        "mask_path": None,
        "atlas_paths_dicts_list": atlases,
        "structure_info_csv_path": tmp_path / "structures_info.csv",
        "tissue_info_csv_path": tmp_path / "tissues_info.csv",
        "save_dir": tmp_path / "out",
        "config": GifConfig(backend="gpu"),
    }


# ----------------------------------------------------------------------------------------------------- configuration
def test_default_backend_is_cpu():
    assert GifConfig().backend == "cpu"


def test_backend_is_validated():
    assert GifConfig(backend="gpu").backend == "gpu"
    with pytest.raises(ValueError, match="backend"):
        GifConfig(backend="tpu")
    with pytest.raises(ValueError, match="gpu_numerics"):
        GifConfig(gpu_numerics=2)
    with pytest.raises(ValueError, match="gpu_batch"):
        GifConfig(gpu_batch=0)


def test_backend_can_be_set_from_strings():
    assert GifConfig().with_overrides({"backend": "gpu", "gpu_numerics": "1"}).gpu_numerics == 1
    assert coerce_field("gpu_write_warped_seg", "false") is False


# ----------------------------------------------------------------------------------------------------- locating the program
def test_missing_binary_message(tmp_path):
    with pytest.raises(BinaryNotFoundError, match="cmake"):
        gpu_backend.find_gpu_binary(GifConfig(gpu_binary=str(tmp_path / "nope")))


def test_binary_from_config_and_environment(tmp_path, monkeypatch):
    binary = make_fake_binary(tmp_path, FAKE_OK)
    assert gpu_backend.find_gpu_binary(GifConfig(gpu_binary=binary)) == binary
    monkeypatch.setenv("GIF_GPU_BINARY", binary)
    assert gpu_backend.find_gpu_binary(GifConfig()) == binary
    monkeypatch.setenv("GIF_GPU_BINARY", str(tmp_path / "nope"))
    with pytest.raises(BinaryNotFoundError):  # an explicit setting is authoritative
        gpu_backend.find_gpu_binary(GifConfig())


def test_missing_binary_is_reported_by_the_pipeline(case, monkeypatch):
    monkeypatch.setattr(gpu_backend, "_CHECKOUT_BINARY", case["save_dir"] / "no_binary")
    monkeypatch.setenv("PATH", "")
    with pytest.raises(BinaryNotFoundError, match="gif_gpu"):
        multi_atlas_segmentation(**case)


# ----------------------------------------------------------------------------------------------------- plan and command line
def test_plan_file(tmp_path):
    structure_df, num_tissue = gpu_backend.read_tables(
        _write(tmp_path / "s.csv", STRUCTURES), _write(tmp_path / "t.csv", TISSUES)
    )
    plan = tmp_path / "plan.txt"
    gpu_backend.write_plan(plan, structure_df, num_tissue)
    assert plan.read_text().splitlines() == ["tissues 3", "structure 0 1 0", "structure 5 1 1", "structure 7 2 1 2"]


def _write(path, text):
    path.write_text(text)
    return path


def test_tissue_missing_in_tissue_info(tmp_path):
    with pytest.raises(ValueError, match="not in the tissue info"):
        gpu_backend.read_tables(
            _write(tmp_path / "s.csv", "label,name,tissues\n0,A,[0]\n1,B,[5]\n"),
            _write(tmp_path / "t.csv", TISSUES),
        )


def test_command_line_carries_the_configuration(tmp_path):
    config = GifConfig(backend="gpu", reg_f3d_be=0.01, seg_em_mrf_beta=0.2, lncc_sigma=(-3, -3, -3), gpu_numerics=1)
    atlases = [{"name": "a", "img_path": "a.nii.gz", "seg_path": "a_seg.nii.gz"}]
    cmd = gpu_backend.build_command("gif_gpu", "t.nii.gz", atlases, "plan.txt", "s.csv", tmp_path, config)
    assert cmd[cmd.index("--f3d-be") + 1] == "0.01"
    assert cmd[cmd.index("--em-mrf-beta") + 1] == "0.2"
    assert cmd[cmd.index("--lncc-sigma") + 1 : cmd.index("--lncc-sigma") + 4] == ["-3.0", "-3.0", "-3.0"]
    assert cmd[cmd.index("--numerics") + 1] == "1"
    assert cmd[cmd.index("--atlas") + 1 : cmd.index("--atlas") + 4] == ["a", "a.nii.gz", "a_seg.nii.gz"]


# ----------------------------------------------------------------------------------------------------- unsupported features
def test_target_mask_is_not_supported(case, write_nifti, tmp_path):
    case["mask_path"] = write_nifti(np.ones((8, 8, 8), np.uint8), tmp_path / "mask.nii.gz")
    with pytest.raises(NotImplementedError, match="mask of the target"):
        multi_atlas_segmentation(**case)


def test_atlas_mask_is_not_supported(case, write_nifti, tmp_path):
    case["atlas_paths_dicts_list"][0]["mask_path"] = write_nifti(np.ones((8, 8, 8), np.uint8), tmp_path / "am.nii.gz")
    with pytest.raises(NotImplementedError, match="atlas masks"):
        multi_atlas_segmentation(**case)


def test_other_unsupported_parameters(case):
    case["config"] = GifConfig(backend="gpu", resample_method=1)
    with pytest.raises(NotImplementedError, match="resample_method"):
        multi_atlas_segmentation(**case)
    case["config"] = GifConfig(backend="gpu", reg_f3d_interp=3)
    with pytest.raises(NotImplementedError, match="linear"):
        multi_atlas_segmentation(**case)


def test_missing_input_files_are_reported_before_the_program_is_needed(case):
    case["img_path"] = "does_not_exist.nii.gz"
    with pytest.raises(FileNotFoundError, match="Input image"):
        multi_atlas_segmentation(**case)


# ----------------------------------------------------------------------------------------------------- running the program
def test_run_with_stand_in_executable(case, tmp_path):
    case["config"] = GifConfig(backend="gpu", gpu_binary=make_fake_binary(tmp_path, FAKE_OK), reg_f3d_be=0.02)
    result = multi_atlas_segmentation(**case)
    assert result.shape == (8, 8, 8) and (result == 7).all()
    argv = json.loads((case["save_dir"] / "argv.json").read_text())
    assert argv[argv.index("--f3d-be") + 1] == "0.02"
    assert argv.count("--atlas") == 2
    assert "structure 7 2 1 2" in (case["save_dir"] / "plan_seen.txt").read_text()


def test_labels_missing_in_structure_info_give_a_value_error(case, tmp_path):
    binary = make_fake_binary(
        tmp_path,
        """
        import sys
        a = sys.argv[1:]
        print("LABELS_NOT_IN_STRUCTURE_INFO\\t/x/seg.nii.gz\\t/x/structures.csv\\t3, 9", file=sys.stderr)
        sys.exit(3)
        """,
    )
    case["config"] = GifConfig(backend="gpu", gpu_binary=binary)
    with pytest.raises(
        ValueError, match=r"/x/seg.nii.gz, but not in the structure info csv file /x/structures.csv: \[3, 9\]"
    ):
        multi_atlas_segmentation(**case)


def test_no_cuda_device_message(case, tmp_path):
    binary = make_fake_binary(
        tmp_path, "import sys\nprint('gif_gpu: no CUDA device available', file=sys.stderr)\nsys.exit(5)\n"
    )
    case["config"] = GifConfig(backend="gpu", gpu_binary=binary)
    with pytest.raises(ExternalToolError, match="CUDA"):
        multi_atlas_segmentation(**case)


def test_failure_of_the_program_reports_the_tail_of_its_output(case, tmp_path):
    binary = make_fake_binary(tmp_path, "import sys\nprint('something broke', file=sys.stderr)\nsys.exit(1)\n")
    case["config"] = GifConfig(backend="gpu", gpu_binary=binary)
    with pytest.raises(ExternalToolError, match="something broke"):
        multi_atlas_segmentation(**case)


def test_unsupported_feature_reported_by_the_program(case, tmp_path):
    binary = make_fake_binary(
        tmp_path, "import sys\nprint('gif_gpu: too many atlases', file=sys.stderr)\nsys.exit(4)\n"
    )
    case["config"] = GifConfig(backend="gpu", gpu_binary=binary)
    with pytest.raises(NotImplementedError, match="too many atlases"):
        multi_atlas_segmentation(**case)


# ----------------------------------------------------------------------------------------------------- command line interface
def test_cli_backend_option(tmp_path, capsys):
    fake = make_fake_binary(tmp_path, FAKE_OK)
    assert main(["segment", "--backend", "gpu", "--gpu-binary", fake, "--print-config"]) == 0
    out = capsys.readouterr().out
    assert '"backend": "gpu"' in out and fake in out
    assert main(["segment", "--print-config"]) == 0
    assert '"backend": "cpu"' in capsys.readouterr().out


def test_cli_segment_with_gpu_backend(case, tmp_path):
    fake = make_fake_binary(tmp_path, FAKE_OK)
    atlas_dir = tmp_path / "atlases"
    for atlas in case["atlas_paths_dicts_list"]:
        folder = atlas_dir / atlas["name"]
        folder.mkdir(parents=True)
        os.symlink(atlas["img_path"], folder / "img.nii.gz")
        os.symlink(atlas["seg_path"], folder / "seg.nii.gz")
    (atlas_dir / "structures_info.csv").write_text(STRUCTURES)
    (atlas_dir / "tissues_info.csv").write_text(TISSUES)
    out = tmp_path / "cli_out"
    status = main(
        ["segment", "--backend", "gpu", "--gpu-binary", fake, "--img", case["img_path"], "--atlas-dir", str(atlas_dir)]
        + ["--atlas-img", "img.nii.gz", "--atlas-seg", "seg.nii.gz", "--out", str(out)]
    )
    assert status == 0 and (out / "final_parcellation.nii.gz").is_file()


# ----------------------------------------------------------------------------------------------------- end to end (needs CUDA)
def _phantom(shift, scale, seed, shape=(96, 96, 96)):
    """Nested ellipsoids (white matter, grey matter, ventricle, two small nuclei) with a smooth bias field and noise.

    :return: image (float32) and label map (uint8: 0 background, 1/2 white matter left/right, 3/4 cortex left/right,
        5 ventricle, 6/7 nucleus left/right)
    """
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape], indexing="ij"), axis=-1)
    centre = np.array(shape, np.float32) / 2 + np.array(shift, np.float32)
    p = (grid - centre) / (np.array([34.0, 40.0, 30.0], np.float32) * scale)
    r = np.sqrt((p**2).sum(-1))
    labels = np.zeros(shape, np.uint8)
    brain = r < 1.0
    wm = r < 0.68
    left = p[..., 0] < 0
    labels[brain & ~wm & left] = 3
    labels[brain & ~wm & ~left] = 4
    labels[wm & left] = 1
    labels[wm & ~left] = 2
    vent = (((p[..., 0]) / 0.18) ** 2 + ((p[..., 1] + 0.05) / 0.35) ** 2 + (p[..., 2] / 0.2) ** 2) < 1
    labels[vent] = 5
    for sign, lab in ((-1, 6), (1, 7)):
        nuc = ((p[..., 0] - sign * 0.3) ** 2 + (p[..., 1] - 0.1) ** 2 + p[..., 2] ** 2) < 0.2**2
        labels[nuc] = lab
    intensity = np.array([5, 150, 150, 100, 100, 25, 125, 125], np.float32)[labels]
    texture = 8.0 * np.sin(grid[..., 0] / 5.0) * np.sin(grid[..., 1] / 6.0) * np.cos(grid[..., 2] / 7.0)
    bias = 1.0 + 0.1 * (grid[..., 0] / shape[0] - 0.5)
    image = (intensity + texture * (labels > 0)) * bias + rng.normal(0, 3.0, shape)
    return image.astype(np.float32), labels


@pytest.mark.gpu
def test_gpu_end_to_end_on_a_phantom(tmp_path):
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    structures = "label,name,tissues\n" + "".join(
        f"{lab},S{lab},[{t}]\n" for lab, t in ((0, 0), (1, 3), (2, 3), (3, 2), (4, 2), (5, 1), (6, 4), (7, 4))
    )
    tissues = "label,name\n0,Other\n1,Csf\n2,Gm\n3,Wm\n4,Deep\n"
    (tmp_path / "structures_info.csv").write_text(structures)
    (tmp_path / "tissues_info.csv").write_text(tissues)
    atlases = []
    for k, (shift, scale) in enumerate([((1, 0, 0), 1.0), ((-2, 1, 0), 1.04), ((0, -2, 1), 0.96), ((2, 2, -1), 1.02)]):
        img, lab = _phantom(shift, scale, seed=k)
        d = tmp_path / f"atlas{k}"
        d.mkdir()
        nib.save(nib.Nifti1Image(img, affine), str(d / "img.nii.gz"))
        nib.save(nib.Nifti1Image(lab, affine), str(d / "seg.nii.gz"))
        atlases.append({"name": f"atlas{k}", "img_path": str(d / "img.nii.gz"), "seg_path": str(d / "seg.nii.gz")})
    target, truth = _phantom((0, 1, -1), 1.0, seed=99)
    nib.save(nib.Nifti1Image(target, affine), str(tmp_path / "target.nii.gz"))
    config = GifConfig(backend="gpu", reg_f3d_maxit=100)
    result = multi_atlas_segmentation(
        str(tmp_path / "target.nii.gz"),
        None,
        atlases,
        tmp_path / "structures_info.csv",
        tmp_path / "tissues_info.csv",
        tmp_path / "out",
        config,
    )
    assert result.shape == truth.shape and result.dtype == np.int32
    for name in (
        "multi_atlas_proba_seg_argmax",
        "multi_atlas_tissue_prior",
        "multi_atlas_tissue_seg",
        "final_parcellation",
    ):
        assert (tmp_path / "out" / f"{name}.nii.gz").is_file()
    assert (tmp_path / "out" / "registered_atlases" / "atlas0" / "warped_atlas_seg.nii.gz").is_file()
    prior = np.asarray(nib.load(str(tmp_path / "out" / "multi_atlas_tissue_prior.nii.gz")).dataobj)
    assert prior.shape == truth.shape + (5,)
    assert np.allclose(prior.sum(-1), 1.0, atol=1e-5)
    dice = []
    for lab in range(1, 8):
        a, b = result == lab, truth == lab
        dice.append(2 * (a & b).sum() / (a.sum() + b.sum()))
    assert np.mean(dice) > 0.8, dice
    # the parcellation is consistent with the tissue segmentation: labels only where their tissue is
    assert set(np.unique(result)) <= set(range(8))
    assert pd.read_csv(tmp_path / "structures_info.csv").shape[0] == 8
