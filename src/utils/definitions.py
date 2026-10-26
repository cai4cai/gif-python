"""Backwards-compatible view of the configuration defaults.

Before the introduction of :class:`src.utils.config.GifConfig` all parameters were module-level
globals defined here. They are still importable under their old names::

    from src.utils.definitions import reg_f3d_LNCC, OMP

but they are only a *read-only view of the defaults* of ``GifConfig`` (computed on first access,
nothing happens at import time). New code should create a ``GifConfig`` and pass it as the
``config`` argument of the pipeline functions.

For old scripts that assign to these names before calling the pipeline (``definitions.OMP = 4``),
the assigned value is picked up by :func:`src.utils.config.default_config`, i.e. by every call
that is made without an explicit ``config``.
"""

from __future__ import annotations

import os
from typing import Any, Dict

# legacy name -> GifConfig field
_LEGACY_TO_FIELD = {
    "reg_aladin_LP": "reg_aladin_lp",
    "reg_f3d_JL": "reg_f3d_jl",
    "reg_f3d_BE": "reg_f3d_be",
    "reg_f3d_MAXIT": "reg_f3d_maxit",
    "reg_f3d_LN": "reg_f3d_ln",
    "reg_f3d_LP": "reg_f3d_lp",
    "reg_f3d_GRID_SPACING": "reg_f3d_grid_spacing",
    "reg_f3d_LNCC": "reg_f3d_lncc",
    "reg_f3d_INTERP": "reg_f3d_interp",
    "RESAMPLE_METHOD": "resample_method",
    "SIGMA": "sigma",
    "LNCC_SIGMA": "lncc_sigma",
    "WEIGHTS_TEMPERATURE": "weights_temperature",
    "seg_EM_MAXIT": "seg_em_maxit",
    "seg_EM_MINIT": "seg_em_minit",
    "seg_EM_BIAS_ORDER": "seg_em_bias_order",
    "seg_EM_BIAS_THRESH": "seg_em_bias_thresh",
    "seg_EM_MRF_BETA": "seg_em_mrf_beta",
    "USE_OLD_RESULTS": "use_old_results",
    "MULTIPROCESSING": "multiprocessing",
    "NUM_POOLS": "num_pools",
    "OMP": "omp",
    "NIFTYREG_PATH": "niftyreg_path",
    "NIFTYSEG_PATH": "niftyseg_path",
}

__all__ = ["ROOT_DIR", *_LEGACY_TO_FIELD]

#: Root of the source checkout (kept for old scripts; meaningless for an installed package).
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def __getattr__(name: str) -> Any:
    # Called only for names that are not (re)assigned in this module.
    if name in _LEGACY_TO_FIELD:
        from src.utils.config import GifConfig

        config = GifConfig()
        if name == "NIFTYREG_PATH":
            return config.niftyreg_dir()
        if name == "NIFTYSEG_PATH":
            return config.niftyseg_dir()
        value = getattr(config, _LEGACY_TO_FIELD[name])
        return list(value) if name == "LNCC_SIGMA" else value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | set(_LEGACY_TO_FIELD))


def legacy_overrides() -> Dict[str, Any]:
    """Values that were explicitly assigned to the legacy globals (name of GifConfig field -> value)."""
    namespace = globals()
    return {field: namespace[name] for name, field in _LEGACY_TO_FIELD.items() if name in namespace}
