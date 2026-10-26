import pytest

from src.utils import definitions
from src.utils.config import (
    BinaryNotFoundError,
    GifConfig,
    coerce_field,
    default_config,
    field_names,
    find_binary,
)

# The values of src/utils/definitions.py before the introduction of GifConfig. They must not change
# (changing a default changes the results of the algorithm).
OLD_DEFINITIONS = {
    "reg_aladin_LP": 2,
    "reg_f3d_JL": 0.0001,
    "reg_f3d_BE": 0.005,
    "reg_f3d_MAXIT": 250,
    "reg_f3d_LN": 4,
    "reg_f3d_LP": 3,
    "reg_f3d_GRID_SPACING": -5.0,
    "reg_f3d_LNCC": 5.0,
    "reg_f3d_INTERP": 1,
    "RESAMPLE_METHOD": 0,
    "SIGMA": 0,
    "LNCC_SIGMA": [-2.5, -2.5, -2.5],
    "WEIGHTS_TEMPERATURE": 0.15,
    "seg_EM_MAXIT": 30,
    "seg_EM_MINIT": 3,
    "seg_EM_BIAS_ORDER": 4,
    "seg_EM_BIAS_THRESH": 0.05,
    "seg_EM_MRF_BETA": 0.1,
    "USE_OLD_RESULTS": True,
    "MULTIPROCESSING": True,
    "NUM_POOLS": 2,
    "OMP": 8,
}


@pytest.mark.parametrize("name,value", sorted(OLD_DEFINITIONS.items()))
def test_legacy_definitions_have_old_values(name, value):
    assert getattr(definitions, name) == value


@pytest.mark.parametrize("name,value", sorted(OLD_DEFINITIONS.items()))
def test_gifconfig_defaults_equal_old_definitions(name, value):
    field = definitions._LEGACY_TO_FIELD[name]
    assert (
        list(getattr(GifConfig(), field)) == value if isinstance(value, list) else getattr(GifConfig(), field) == value
    )


def test_every_old_definition_has_a_field():
    assert set(definitions._LEGACY_TO_FIELD) >= set(OLD_DEFINITIONS)
    assert {"niftyreg_path", "niftyseg_path"} <= set(field_names())


def test_import_of_definitions_exposes_old_names():
    from src.utils.definitions import NIFTYREG_PATH, NIFTYSEG_PATH, ROOT_DIR, seg_EM_MRF_BETA

    assert seg_EM_MRF_BETA == 0.1
    assert isinstance(ROOT_DIR, str)
    assert isinstance(NIFTYREG_PATH, str) and isinstance(NIFTYSEG_PATH, str)


def test_unknown_legacy_attribute_raises():
    with pytest.raises(AttributeError):
        definitions.DOES_NOT_EXIST  # noqa: B018


def test_assigning_legacy_global_is_honoured_by_default_config(monkeypatch):
    monkeypatch.setattr(definitions, "OMP", 3, raising=False)
    assert default_config().omp == 3
    assert GifConfig().omp == 8


def test_config_is_immutable_and_hashable():
    config = GifConfig()
    with pytest.raises(Exception):
        config.omp = 1  # type: ignore[misc]
    assert hash(config) == hash(GifConfig())
    assert config.replace(omp=2).omp == 2 and config.omp == 8


def test_lncc_sigma_accepts_lists_and_validates_length():
    assert GifConfig(lncc_sigma=[-2, -2, -2]).lncc_sigma == (-2.0, -2.0, -2.0)
    with pytest.raises(ValueError):
        GifConfig(lncc_sigma=(1.0, 2.0))


@pytest.mark.parametrize("kwargs", [{"num_pools": 0}, {"omp": 0}, {"resample_method": 2}])
def test_invalid_values_rejected(kwargs):
    with pytest.raises(ValueError):
        GifConfig(**kwargs)


def test_with_overrides_converts_strings():
    config = GifConfig().with_overrides(
        {"reg_f3d_lncc": "7.5", "reg_f3d_maxit": "100", "use_old_results": "false", "lncc_sigma": "-3 -3 -3"}
    )
    assert config.reg_f3d_lncc == 7.5
    assert config.reg_f3d_maxit == 100
    assert config.use_old_results is False
    assert config.lncc_sigma == (-3.0, -3.0, -3.0)


def test_unknown_override_lists_valid_names():
    with pytest.raises(ValueError, match="reg_f3d_be"):
        coerce_field("not_a_parameter", "1")


def test_binary_from_explicit_directory(fake_bin_dir):
    config = GifConfig(niftyreg_path=str(fake_bin_dir), niftyseg_path=str(fake_bin_dir))
    assert config.binary("reg_f3d") == str(fake_bin_dir / "reg_f3d")
    assert config.binary("seg_EM") == str(fake_bin_dir / "seg_EM")


def test_binary_from_environment(fake_bin_dir, monkeypatch):
    monkeypatch.setenv("NIFTYREG_PATH", str(fake_bin_dir))
    monkeypatch.setenv("NIFTYSEG_PATH", str(fake_bin_dir))
    config = GifConfig()
    assert config.binary("reg_aladin") == str(fake_bin_dir / "reg_aladin")
    assert config.niftyseg_dir() == str(fake_bin_dir)


def test_explicit_directory_wins_over_environment(fake_bin_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("NIFTYREG_PATH", str(tmp_path / "elsewhere"))
    assert GifConfig(niftyreg_path=str(fake_bin_dir)).binary("reg_f3d") == str(fake_bin_dir / "reg_f3d")


def test_binary_from_path(fake_bin_dir, monkeypatch):
    monkeypatch.setenv("PATH", str(fake_bin_dir))
    assert GifConfig().binary("reg_resample") == str(fake_bin_dir / "reg_resample")


def test_missing_binary_gives_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(BinaryNotFoundError, match="NIFTYREG_PATH"):
        find_binary("reg_aladin_does_not_exist", env_var="NIFTYREG_PATH")


def test_configured_directory_without_binary_is_an_error_not_a_fallback(tmp_path, fake_bin_dir, monkeypatch):
    monkeypatch.setenv("PATH", str(fake_bin_dir))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(BinaryNotFoundError, match="reg_f3d"):
        GifConfig(niftyreg_path=str(empty)).binary("reg_f3d")


def test_config_module_does_no_work_at_import():
    import subprocess
    import sys

    code = (
        "import sys; import src.utils.config, src.utils.definitions, src.multi_atlas.multi_atlas_segmentation; "
        "import nibabel as nib; "
        "print(hasattr(nib.Nifti1Header, 'quaternion_threshold') and nib.Nifti1Header.quaternion_threshold)"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout.strip()
    # nibabel's own default is 1e-6 (a positive number); the workaround sets a negative threshold
    assert float(out) > 0


def test_new_options_default_to_the_previous_behaviour():
    assert GifConfig().keep_registration_files is False
