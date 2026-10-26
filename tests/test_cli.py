import json
import subprocess
import sys

import pytest

from src.cli.main import COMMANDS, build_parser, main
from src.multi_atlas.utils import find_atlases


def test_main_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in COMMANDS:
        assert name in out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "gif-python" in capsys.readouterr().out


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_subcommand_help(command, capsys):
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
    assert f"gif-python {command}" in capsys.readouterr().out


def test_expected_commands_are_registered():
    assert "segment" in COMMANDS
    assert build_parser().prog == "gif-python"


def test_module_entry_point():
    result = subprocess.run([sys.executable, "-m", "src.cli", "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "segment" in result.stdout


def test_command_is_required(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_segment_requires_img_atlas_dir_and_out(capsys):
    assert main(["segment"]) == 1
    assert "--img" in capsys.readouterr().err


def test_segment_print_config(capsys):
    assert main(["segment", "--print-config", "--config", "reg_f3d_lncc=7", "--omp", "3", "--no-multiprocessing"]) == 0
    out = capsys.readouterr().out
    config = json.loads(out[: out.index("\n# ")])
    assert config["reg_f3d_lncc"] == 7.0 and config["omp"] == 3 and config["multiprocessing"] is False
    assert config["seg_em_mrf_beta"] == 0.1


def test_segment_rejects_unknown_config_name(capsys):
    assert main(["segment", "--print-config", "--config", "nonsense=1"]) == 1
    assert "nonsense" in capsys.readouterr().err
    assert main(["segment", "--print-config", "--config", "justaname"]) == 1


def make_atlas_folder(root, names, seg="seg.nii.gz", img="img.nii.gz", incomplete=()):
    for name in names:
        (root / name).mkdir(parents=True)
        (root / name / img).write_text("")
        if name not in incomplete:
            (root / name / seg).write_text("")
    return root


def test_find_atlases_natural_order_and_exclusion(tmp_path):
    root = make_atlas_folder(tmp_path / "atlases", ["a-10", "a-2", "a-1", "a-3"])
    names = [a["name"] for a in find_atlases(root, "img.nii.gz", "seg.nii.gz")]
    assert names == ["a-1", "a-2", "a-3", "a-10"]
    names = [a["name"] for a in find_atlases(root, "img.nii.gz", "seg.nii.gz", exclude=["a-2"])]
    assert names == ["a-1", "a-3", "a-10"]


def test_find_atlases_skips_incomplete_folders_and_fails_without_atlases(tmp_path):
    root = make_atlas_folder(tmp_path / "atlases", ["a", "b"], incomplete=["b"])
    assert [a["name"] for a in find_atlases(root, "img.nii.gz", "seg.nii.gz")] == ["a"]
    with pytest.raises(FileNotFoundError):
        find_atlases(root, "img.nii.gz", "other.nii.gz")
    with pytest.raises(FileNotFoundError):
        find_atlases(tmp_path / "missing", "img.nii.gz", "seg.nii.gz")


def test_segment_passes_arguments_to_the_pipeline(tmp_path, monkeypatch):
    root = make_atlas_folder(tmp_path / "atlases", ["a1", "a2", "target"], seg="labels.nii.gz", img="t1.nii.gz")
    (root / "structures_info.csv").write_text("label,name,tissues\n")
    (root / "tissues_info.csv").write_text("label,name\n")
    calls = []

    import src.multi_atlas.multi_atlas_segmentation as module

    monkeypatch.setattr(module, "multi_atlas_segmentation", lambda **kwargs: calls.append(kwargs))
    code = main(
        [
            "segment",
            "--img", str(tmp_path / "target.nii.gz"),
            "--atlas-dir", str(root),
            "--atlas-img", "t1.nii.gz",
            "--atlas-seg", "labels.nii.gz",
            "--exclude-atlas", "target",
            "--out", str(tmp_path / "out"),
            "--config", "reg_f3d_be=0.01",
            "--no-multiprocessing",
        ]
    )  # fmt: skip
    assert code == 0 and len(calls) == 1
    call = calls[0]
    assert [a["name"] for a in call["atlas_paths_dicts_list"]] == ["a1", "a2"]
    assert call["config"].reg_f3d_be == 0.01 and call["config"].multiprocessing is False
    assert str(call["structure_info_csv_path"]).endswith("structures_info.csv")


def test_segment_skip_existing(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "final_parcellation.nii.gz").write_text("")
    import src.multi_atlas.multi_atlas_segmentation as module

    monkeypatch.setattr(module, "multi_atlas_segmentation", lambda **kwargs: pytest.fail("must not run"))
    code = main(["segment", "--img", "x.nii.gz", "--atlas-dir", str(tmp_path), "--out", str(out), "--skip-existing"])
    assert code == 0


def test_segment_missing_structure_info_is_reported(tmp_path, capsys):
    root = make_atlas_folder(tmp_path / "atlases", ["a1"], seg="labels.nii.gz", img="t1.nii.gz")
    code = main(
        ["segment", "--img", "x.nii.gz", "--atlas-dir", str(root), "--atlas-img", "t1.nii.gz",
         "--atlas-seg", "labels.nii.gz", "--out", str(tmp_path / "o")]
    )  # fmt: skip
    assert code == 1 and "structures_info.csv" in capsys.readouterr().err


@pytest.mark.parametrize("script", ["run_registration_parcellation_basic.py", "run_registration_parcellation_mb42.py"])
def test_old_top_level_scripts_still_start(script):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / script), "--help"], capture_output=True, text=True, cwd=root)
    assert result.returncode == 0, result.stderr
