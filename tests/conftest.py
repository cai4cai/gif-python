"""Shared fixtures. No test needs data on disk: volumes are created in ``tmp_path``."""

from __future__ import annotations

import stat
import subprocess

import nibabel as nib
import numpy as np
import pytest

from src.multi_atlas.gpu_backend import find_gpu_binary
from src.utils.config import BinaryNotFoundError, GifConfig

REQUIRED_BINARIES = ("reg_aladin", "reg_f3d", "reg_resample", "seg_EM")


def _binaries_available() -> bool:
    config = GifConfig()
    try:
        for name in REQUIRED_BINARIES:
            config.binary(name)
    except BinaryNotFoundError:
        return False
    return True


def _gpu_available() -> bool:
    """True if the gif_gpu executable exists and reports a usable CUDA device."""
    try:
        binary = find_gpu_binary(GifConfig())
        return subprocess.run([binary, "--check-device"], capture_output=True).returncode == 0
    except (OSError, BinaryNotFoundError):
        return False


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``integration`` / ``gpu`` when NiftyReg / NiftySeg / the CUDA program are not available."""
    if not _binaries_available():
        skip = pytest.mark.skip(
            reason="NiftyReg/NiftySeg executables not found (put them on PATH or set NIFTYREG_PATH / NIFTYSEG_PATH)"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)
    if any("gpu" in item.keywords for item in items) and not _gpu_available():
        skip = pytest.mark.skip(reason="gif_gpu executable or CUDA device not available (see native/README.md)")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    """Tests must not depend on the user's NiftyReg configuration."""
    monkeypatch.delenv("NIFTYREG_PATH", raising=False)
    monkeypatch.delenv("NIFTYSEG_PATH", raising=False)
    monkeypatch.delenv("GIF_GPU_BINARY", raising=False)


@pytest.fixture
def fake_bin_dir(tmp_path):
    """A directory with executable stand-ins for all NiftyReg / NiftySeg tools."""
    bin_dir = tmp_path / "fake_bin"
    bin_dir.mkdir()
    for name in REQUIRED_BINARIES:
        path = bin_dir / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return bin_dir


def save_nifti(array: np.ndarray, path, affine: np.ndarray | None = None):
    """Write ``array`` as NIfTI to ``path`` (str or Path) and return the path as ``str``."""
    nib.save(nib.Nifti1Image(array, np.eye(4) if affine is None else affine), str(path))
    return str(path)


@pytest.fixture
def write_nifti():
    return save_nifti


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
