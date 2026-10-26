"""Run-time configuration of the multi-atlas segmentation.

All parameters that used to be module-level globals in :mod:`src.utils.definitions` live in the
:class:`GifConfig` dataclass. The defaults are identical to the former globals; they are *not*
the defaults of the C++ GIF implementation in every case (see the notes next to the fields).

Nothing is computed when this module is imported. The locations of the NiftyReg and NiftySeg
binaries are resolved lazily (see :func:`find_binary`).
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#: Environment variables that point to the directories containing the NiftyReg / NiftySeg binaries.
NIFTYREG_ENV_VAR = "NIFTYREG_PATH"
NIFTYSEG_ENV_VAR = "NIFTYSEG_PATH"
#: Location that was hardcoded before; used as the last resort.
LEGACY_BIN_DIR = "/usr/local/bin"
#: values of :attr:`GifConfig.backend`
BACKENDS = ("cpu", "gpu")
#: Environment variable that points to the ``gif_gpu`` executable of the GPU backend.
GIF_GPU_ENV_VAR = "GIF_GPU_BINARY"


class BinaryNotFoundError(FileNotFoundError):
    """A NiftyReg or NiftySeg executable could not be located."""


@dataclass(frozen=True)
class GifConfig:
    """Parameters of the pipeline.

    The configuration is immutable; use :meth:`replace` or :meth:`with_overrides` to derive a
    modified copy. The values below are the former defaults of ``src.utils.definitions`` and are
    unchanged.

    Notes
    -----
    Several defaults differ from the C++ GIF (KCL-BMEIS/gif) implementation, for example the MRF
    weight of the final EM (``seg_em_mrf_beta``: 0.1 here, 0.15 in the C++ ``GIF`` constructor)
    and the LNCC kernel used for the fusion weights (``lncc_sigma``: -2.5 voxels here, -2.0 in
    C++). They are kept as they are; changing them is a separate decision.
    """

    # Linear registration (reg_aladin)
    reg_aladin_lp: int = 2  #: number of levels to run once the pyramids have been created (-lp)

    # Non-linear registration (reg_f3d)
    reg_f3d_jl: float = 0.0001  #: weight of the log-Jacobian determinant penalty (-jl)
    reg_f3d_be: float = 0.005  #: weight of the bending energy penalty (-be)
    reg_f3d_maxit: int = 250  #: maximal number of iterations at the final level (-maxit)
    reg_f3d_ln: int = 4  #: number of levels (-ln)
    reg_f3d_lp: int = 3  #: only the first ``lp`` levels of the pyramid are run (-lp)
    reg_f3d_grid_spacing: float = -5.0  #: final grid spacing (-sx); mm if > 0, voxels if < 0
    reg_f3d_lncc: float = 5.0  #: standard deviation of the Gaussian kernel of the LNCC cost (--lncc)
    reg_f3d_interp: int = 1  #: interpolation order (--interp): 0 = NN, 1 = linear, 3 = cubic

    # Resampling of the atlas segmentations
    #: 0: nearest neighbour on the label map; 1: (smoothing and) linear interpolation of one-hot
    #: labels (not implemented)
    resample_method: int = 0
    #: std. dev. of the Gaussian used to smooth label maps before resampling (method 1 only)
    sigma: float = 0

    # LNCC distance used for the fusion weights
    #: std. dev. of the Gaussian smoothing kernel per axis. A negative value is read as millimetres
    #: and divided by the voxel size of the image; a positive value is used as given, in voxels.
    lncc_sigma: Tuple[float, float, float] = (-2.5, -2.5, -2.5)

    # Weights
    weights_temperature: float = 0.15  #: temperature of the LNCC distance -> weight conversion

    # seg_EM (EM refinement of the tissue prior)
    seg_em_maxit: int = 30  #: maximal number of iterations (-max_iter)
    seg_em_minit: int = 3  #: minimal number of iterations (-min_iter)
    seg_em_bias_order: int = 4  #: order of the bias field (-bc_order)
    seg_em_bias_thresh: float = 0.05  #: threshold to stop the bias field estimation (-bc_thresh)
    seg_em_mrf_beta: float = 0.1  #: weight of the MRF prior (-mrf_beta)

    # Re-use of intermediate results (registration, resampling, weights) found in the output folder
    use_old_results: bool = True
    #: keep the affine, control point grid and warped atlas images in registered_atlases/<name>/ (they are
    #: deleted after use by default), e.g. to inspect the registration quality
    keep_registration_files: bool = False

    # Parallelism
    multiprocessing: bool = True  #: process several atlases in parallel
    num_pools: int = 2  #: number of atlases processed in parallel (more pools need more RAM)
    omp: int = 8  #: OpenMP threads of each NiftyReg call (``omp * num_pools`` <= cores)

    # Execution backend: "cpu" (NiftyReg / NiftySeg executables) or "gpu" (CUDA program of native/, native/README.md)
    backend: str = "cpu"
    #: path of the ``gif_gpu`` executable of the GPU backend. ``None`` = environment variable GIF_GPU_BINARY, then
    #: ``native/build/gif_gpu`` of the source checkout, then PATH
    gpu_binary: Optional[str] = None
    #: GPU backend only: write ``registered_atlases/<atlas>/warped_atlas_seg.nii.gz`` (as the CPU backend does)
    gpu_write_warped_seg: bool = True
    #: GPU backend only: number of atlases that are registered together by the non-rigid registration (0 = automatic,
    #: about 6 GB of GPU memory; lower it to save memory)
    gpu_batch: int = 0
    #: GPU backend only: numerics of the non-rigid registration: 0 = exact emulation of NiftyReg's arithmetic,
    #: 1 = validated fast option (about 2x faster, same quality within noise; see native/README.md)
    gpu_numerics: int = 0

    # Locations of the NiftyReg / NiftySeg binaries. ``None`` = look at the environment variable
    # (NIFTYREG_PATH / NIFTYSEG_PATH), then at PATH, then at /usr/local/bin.
    niftyreg_path: Optional[str] = None  #: directory with reg_aladin, reg_f3d, reg_resample
    niftyseg_path: Optional[str] = None  #: directory with seg_EM

    def __post_init__(self) -> None:
        # accept lists (e.g. from YAML/JSON/CLI) and keep the dataclass hashable
        object.__setattr__(self, "lncc_sigma", tuple(float(v) for v in self.lncc_sigma))
        if len(self.lncc_sigma) != 3:
            raise ValueError(f"lncc_sigma needs three values (x, y, z), got {self.lncc_sigma!r}")
        if self.num_pools < 1:
            raise ValueError(f"num_pools must be >= 1, got {self.num_pools}")
        if self.omp < 1:
            raise ValueError(f"omp must be >= 1, got {self.omp}")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {', '.join(BACKENDS)}, got {self.backend!r}")
        if self.gpu_numerics not in (0, 1):
            raise ValueError(f"gpu_numerics must be 0 or 1, got {self.gpu_numerics}")
        if self.gpu_batch < 0:
            raise ValueError(f"gpu_batch must be >= 0 (0 = automatic), got {self.gpu_batch}")
        if self.resample_method not in (0, 1):
            raise ValueError(f"resample_method must be 0 or 1, got {self.resample_method}")

    # ------------------------------------------------------------------ derived copies
    def replace(self, **changes: Any) -> GifConfig:
        """Return a copy with the given fields changed."""
        return dataclasses.replace(self, **changes)

    def with_overrides(self, overrides: Optional[Dict[str, Any]] = None, **kwargs: Any) -> GifConfig:
        """Like :meth:`replace`, but values may be strings (``"0.2"``) that are converted to the
        type of the field, and unknown names raise a ``ValueError`` listing the valid ones."""
        merged = dict(overrides or {}, **kwargs)
        return self.replace(**{k: coerce_field(k, v) for k, v in merged.items()})

    def to_dict(self) -> Dict[str, Any]:
        """All parameters as a plain dictionary."""
        return dataclasses.asdict(self)

    # ------------------------------------------------------------------ binaries
    def niftyreg_dir(self) -> str:
        """Directory in which the NiftyReg executables are looked up (see :func:`find_binary`)."""
        return _tool_dir(self.niftyreg_path, NIFTYREG_ENV_VAR, "reg_aladin")

    def niftyseg_dir(self) -> str:
        """Directory in which the NiftySeg executables are looked up (see :func:`find_binary`)."""
        return _tool_dir(self.niftyseg_path, NIFTYSEG_ENV_VAR, "seg_EM")

    def binary(self, name: str) -> str:
        """Full path of the NiftyReg/NiftySeg executable ``name``.

        Raises :class:`BinaryNotFoundError` with a hint on how to fix it if it is not available.
        """
        if name.startswith("seg_"):
            return find_binary(name, self.niftyseg_path, NIFTYSEG_ENV_VAR)
        return find_binary(name, self.niftyreg_path, NIFTYREG_ENV_VAR)


def _tool_dir(explicit: Optional[str], env_var: str, probe: str) -> str:
    """Directory that is searched for the tool family of ``probe`` (no existence check)."""
    if explicit:
        return str(explicit)
    if os.environ.get(env_var):
        return os.environ[env_var]
    found = shutil.which(probe)
    if found:
        return str(Path(found).parent)
    return LEGACY_BIN_DIR


def find_binary(name: str, directory: Optional[str] = None, env_var: Optional[str] = None) -> str:
    """Locate the executable ``name``.

    Order: ``directory`` (explicit setting), the directory in the environment variable ``env_var``,
    the executable on ``PATH``, and finally ``/usr/local/bin`` (the location that used to be hard
    coded). An explicit directory or environment variable is authoritative: if it is set but does
    not contain the executable an error is raised instead of silently using another copy.
    """
    candidates = []
    if directory:
        candidates.append((Path(directory) / name, "the configured directory"))
    elif env_var and os.environ.get(env_var):
        candidates.append((Path(os.environ[env_var]) / name, f"${env_var}"))
    if candidates:
        path, origin = candidates[0]
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        raise BinaryNotFoundError(
            f"{name} not found or not executable at {path} (from {origin}). "
            f"Set {env_var or 'the directory'} to the folder that contains {name}."
        )
    on_path = shutil.which(name)
    if on_path:
        return on_path
    legacy = Path(LEGACY_BIN_DIR) / name
    if legacy.is_file() and os.access(legacy, os.X_OK):
        return str(legacy)
    raise BinaryNotFoundError(
        f"{name} was not found. Install NiftyReg/NiftySeg and make it available on PATH, or set the "
        f"{env_var or 'NIFTYREG_PATH / NIFTYSEG_PATH'} environment variable to the directory that "
        f"contains it (see INSTALL.md)."
    )


def _field_types() -> Dict[str, Any]:
    return typing.get_type_hints(GifConfig)


def field_names() -> Tuple[str, ...]:
    """Names of all configuration parameters."""
    return tuple(f.name for f in dataclasses.fields(GifConfig))


def coerce_field(name: str, value: Any) -> Any:
    """Convert ``value`` (possibly a string from the command line) to the type of field ``name``."""
    types = _field_types()
    if name not in types:
        raise ValueError(f"Unknown configuration parameter {name!r}. Valid parameters: {', '.join(field_names())}")
    if not isinstance(value, str):
        return value
    tp = types[name]
    text = value.strip()
    if tp is bool:
        if text.lower() in ("1", "true", "yes", "on"):
            return True
        if text.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"Cannot interpret {value!r} as a boolean for {name}")
    if tp is int:
        return int(text)
    if tp is float:
        return float(text)
    if typing.get_origin(tp) is tuple:
        return tuple(float(v) for v in text.strip("()[]").replace(",", " ").split())
    if text.lower() in ("none", ""):
        return None
    return text


def default_config() -> GifConfig:
    """The configuration used when a function is called without ``config``.

    Equal to ``GifConfig()``, except that values assigned to the legacy module globals in
    :mod:`src.utils.definitions` (for example ``definitions.OMP = 4`` in an old script) are
    honoured, so existing scripts keep working.
    """
    from src.utils import definitions

    return GifConfig().replace(**definitions.legacy_overrides())
