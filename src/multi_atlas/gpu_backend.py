"""GPU backend: runs the multi-atlas pipeline in the CUDA program ``gif_gpu`` (built from ``native/``).

The program takes the same inputs as :func:`src.multi_atlas.multi_atlas_segmentation.multi_atlas_segmentation` (it is
given the structure/tissue tables as a small text "plan") and writes the same output files. Nothing CUDA related is
imported here: the
executable is only looked up and started when the backend is used.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from ast import literal_eval
from pathlib import Path
from typing import Dict, List, Optional, Union

import nibabel as nib
import numpy as np
import pandas as pd

from src.utils.config import GIF_GPU_ENV_VAR, BinaryNotFoundError, GifConfig, default_config
from src.utils.process import ExternalToolError

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

#: ``native/build/gif_gpu`` of a source checkout (the default output of the CMake build described in native/README.md)
_CHECKOUT_BINARY = Path(__file__).resolve().parents[2] / "native" / "build" / "gif_gpu"

#: marker that the program prints when atlas labels are missing in the structure info (exit status 3)
_LABELS_MARKER = "LABELS_NOT_IN_STRUCTURE_INFO"

_BUILD_HINT = (
    "Build it with: cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release && cmake --build native/build -j "
    "(needs the CUDA toolkit, see native/README.md), or point to it with --gpu-binary / "
    f"the {GIF_GPU_ENV_VAR} environment variable / the gpu_binary configuration parameter."
)


def find_gpu_binary(config: Optional[GifConfig] = None) -> str:
    """Locate the ``gif_gpu`` executable.

    Order: ``config.gpu_binary``, the environment variable ``GIF_GPU_BINARY``, ``native/build/gif_gpu`` of the source
    checkout, ``PATH``.
    An explicit setting is authoritative (no silent fall back to another copy).

    :raises BinaryNotFoundError: if it cannot be found; the message explains how to build it
    """
    config = config or default_config()
    explicit = config.gpu_binary or os.environ.get(GIF_GPU_ENV_VAR)
    if explicit:
        path = Path(explicit)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        raise BinaryNotFoundError(f"gif_gpu not found or not executable at {path}. {_BUILD_HINT}")
    if _CHECKOUT_BINARY.is_file() and os.access(_CHECKOUT_BINARY, os.X_OK):
        return str(_CHECKOUT_BINARY)
    on_path = shutil.which("gif_gpu")
    if on_path:
        return on_path
    raise BinaryNotFoundError(f"The GPU backend needs the gif_gpu executable, which was not found. {_BUILD_HINT}")


def _check_supported(
    mask_path: Optional[PathLike], atlas_paths_dicts_list: List[Dict[str, str]], config: GifConfig
) -> None:
    """The GPU backend does not implement every feature of the CPU backend: say so clearly instead of ignoring it."""
    if mask_path:
        raise NotImplementedError(
            "backend 'gpu' does not support a mask of the target image (-rmask of reg_aladin/reg_f3d, mask of "
            "seg_EM and of the LNCC weights). Run without a mask or use backend 'cpu'."
        )
    if any(a.get("mask_path") for a in atlas_paths_dicts_list):
        raise NotImplementedError(
            "backend 'gpu' does not support atlas masks (-fmask of reg_aladin/reg_f3d). Run without atlas masks or use "
            "backend 'cpu'."
        )
    if config.resample_method != 0:
        raise NotImplementedError(
            f"resample_method={config.resample_method} is not implemented; only 0 (nearest neighbour) is supported"
        )
    if config.reg_f3d_interp != 1:
        raise NotImplementedError(
            f"backend 'gpu' only supports linear interpolation in reg_f3d (reg_f3d_interp=1), "
            f"got {config.reg_f3d_interp}"
        )


def _require_file(path: PathLike, what: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{what} does not exist: {path}")


def read_tables(structure_info_csv_path: PathLike, tissue_info_csv_path: PathLike):
    """Read and check the structure and tissue tables like the CPU backend does.

    :return: ``(structure_df, num_tissue)``; ``structure_df`` is indexed by label and has a ``tissues`` column
        (list of int)
    :raises ValueError: if the structure info mentions tissues that are not in the tissue info
    """
    tissue_dict = pd.read_csv(tissue_info_csv_path, index_col="label").to_dict(orient="dict")
    num_tissue = len(tissue_dict["name"])
    structure_df = pd.read_csv(structure_info_csv_path, index_col="label", converters={"tissues": literal_eval})
    unique_tissues = np.unique(np.concatenate(structure_df["tissues"].values))
    missing = set(unique_tissues) - set(tissue_dict["name"].keys())
    if missing:
        raise ValueError(
            f"The following tissues are present in the structure info csv file {structure_info_csv_path}, "
            f"but not in the tissue info csv file {tissue_info_csv_path}: {sorted(int(t) for t in missing)}."
        )
    return structure_df, num_tissue


def write_plan(path: PathLike, structure_df: pd.DataFrame, num_tissue: int) -> None:
    """Write the structure/tissue tables in the line format that ``gif_gpu`` reads."""
    lines = [f"tissues {int(num_tissue)}"]
    for label, tissues in structure_df["tissues"].items():
        tissues = [int(t) for t in tissues]
        lines.append(f"structure {int(label)} {len(tissues)} {' '.join(str(t) for t in tissues)}".rstrip())
    Path(path).write_text("\n".join(lines) + "\n")


def build_command(
    binary: str,
    img_path: PathLike,
    atlas_paths_dicts_list: List[Dict[str, str]],
    plan_path: PathLike,
    structure_info_csv_path: PathLike,
    save_dir: PathLike,
    config: GifConfig,
) -> List[str]:
    """Command line of ``gif_gpu`` for the given inputs and configuration."""
    cmd = [
        binary,
        "--img", str(img_path),
        "--out", str(save_dir),
        "--plan", str(plan_path),
        "--structure-info", str(structure_info_csv_path),
        "--aladin-lp", str(config.reg_aladin_lp),
        "--f3d-jl", repr(config.reg_f3d_jl),
        "--f3d-be", repr(config.reg_f3d_be),
        "--f3d-maxit", str(config.reg_f3d_maxit),
        "--f3d-ln", str(config.reg_f3d_ln),
        "--f3d-lp", str(config.reg_f3d_lp),
        "--f3d-sx", repr(config.reg_f3d_grid_spacing),
        "--f3d-lncc", repr(config.reg_f3d_lncc),
        "--f3d-interp", str(config.reg_f3d_interp),
        "--lncc-sigma", *(repr(v) for v in config.lncc_sigma),
        "--temperature", repr(config.weights_temperature),
        "--em-maxit", str(config.seg_em_maxit),
        "--em-minit", str(config.seg_em_minit),
        "--em-bias-order", str(config.seg_em_bias_order),
        "--em-bias-thresh", repr(config.seg_em_bias_thresh),
        "--em-mrf-beta", repr(config.seg_em_mrf_beta),
        "--numerics", str(config.gpu_numerics),
        "--f3d-batch", str(config.gpu_batch),
        "--write-warped-seg", "1" if config.gpu_write_warped_seg else "0",
    ]  # fmt: skip
    for atlas in atlas_paths_dicts_list:
        cmd += ["--atlas", str(atlas["name"]), str(atlas["img_path"]), str(atlas["seg_path"])]
    return cmd


def _raise_for_status(result: subprocess.CompletedProcess) -> None:
    """Translate the exit status of ``gif_gpu`` into the exceptions of the CPU backend."""
    if result.returncode == 0:
        return
    stderr = result.stderr or ""
    tail = "\n".join((stderr.strip() or (result.stdout or "").strip()).splitlines()[-20:])
    for line in stderr.splitlines():
        if line.startswith(_LABELS_MARKER):
            _, seg_path, csv_path, labels = (line.split("\t") + ["", "", "", ""])[:4]
            label_list = [int(x) for x in re.findall(r"\d+", labels)]
            raise ValueError(
                f"The following labels are present in the atlas segmentation {seg_path}, "
                f"but not in the structure info csv file {csv_path}: {label_list}."
            )
    if result.returncode == 4:
        raise NotImplementedError(f"backend 'gpu': {tail}")
    if result.returncode == 5:
        raise ExternalToolError(
            f"backend 'gpu' needs a CUDA capable GPU and driver, but no CUDA device is available.\n{tail}"
        )
    raise ExternalToolError(f"gif_gpu failed with exit status {result.returncode}.\nLast output:\n{tail}")


def multi_atlas_segmentation_gpu(
    img_path: PathLike,
    mask_path: Optional[PathLike],
    atlas_paths_dicts_list: List[Dict[str, str]],
    structure_info_csv_path: PathLike,
    tissue_info_csv_path: PathLike,
    save_dir: PathLike,
    config: Optional[GifConfig] = None,
) -> np.ndarray:
    """GPU version of :func:`~src.multi_atlas.multi_atlas_segmentation.multi_atlas_segmentation` (same arguments).

    Writes ``multi_atlas_proba_seg_argmax.nii.gz``, ``multi_atlas_tissue_prior.nii.gz``,
    ``multi_atlas_tissue_seg.nii.gz``, ``final_parcellation.nii.gz`` and (if ``config.gpu_write_warped_seg``)
    ``registered_atlases/<name>/warped_atlas_seg.nii.gz`` into ``save_dir`` and returns the final parcellation.

    :raises NotImplementedError: for features that the GPU backend does not support (masks)
    :raises BinaryNotFoundError: if the ``gif_gpu`` executable is missing
    :raises ExternalToolError: if no CUDA device is available or ``gif_gpu`` fails
    :raises FileNotFoundError: if an input file does not exist
    :raises ValueError: if the structure/tissue information is inconsistent with the atlases
    """
    config = config or default_config()
    time_0 = time.time()
    logger.info("Multi-atlas segmentation using the GIF-like fusion method (GPU backend)...")
    _require_file(img_path, "Input image")
    if mask_path is not None:
        _require_file(mask_path, "Input mask")
    atlas_paths_dicts_list = [dict(a) for a in atlas_paths_dicts_list]
    for atlas in atlas_paths_dicts_list:
        _require_file(atlas["img_path"], "Atlas image")
        _require_file(atlas["seg_path"], "Atlas segmentation")
        if "mask_path" in atlas and atlas["mask_path"]:
            _require_file(atlas["mask_path"], "Atlas mask")
    _require_file(structure_info_csv_path, "Structure info csv file")
    _require_file(tissue_info_csv_path, "Tissue info csv file")
    _check_supported(mask_path, atlas_paths_dicts_list, config)
    structure_df, num_tissue = read_tables(structure_info_csv_path, tissue_info_csv_path)
    binary = find_gpu_binary(config)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gif_gpu_") as tmp:
        plan_path = Path(tmp) / "plan.txt"
        write_plan(plan_path, structure_df, num_tissue)
        cmd = build_command(
            binary, img_path, atlas_paths_dicts_list, plan_path, structure_info_csv_path, save_dir, config
        )
        logger.debug("Running: %s", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as exc:
            raise ExternalToolError(f"Cannot execute {binary!r}: {exc}") from exc
    _raise_for_status(result)
    if result.stdout.strip():
        logger.debug("gif_gpu output:\n%s", result.stdout.rstrip())
    logger.info("GPU multi-atlas segmentation completed after %.3f seconds", time.time() - time_0)
    return np.asarray(nib.load(str(save_dir / "final_parcellation.nii.gz")).dataobj)
