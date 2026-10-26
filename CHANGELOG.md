# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `pyproject.toml` (PEP 621) with all dependencies, `dev` and `viz` extras and the console script `gif-python`.
  The distribution name (`pygif`) and the import path (`src`) are unchanged.
- `gif-python` command with `segment`, `clean-labels`, `make-structure-info` and `fetch-mindboggle`.
- `GifConfig` (`src.utils.config`): all pipeline parameters in one immutable dataclass, passed as the optional
  `config` argument of the pipeline functions. Defaults are the values of the former `definitions.py`.
  NiftyReg/NiftySeg are found through `NIFTYREG_PATH` / `NIFTYSEG_PATH`, `PATH`, then `/usr/local/bin`.
- `keep_registration_files` option / `--keep-registration-files` to keep the registration results for inspection.
- Mindboggle-101 label tables (`structures_info.csv` with all 114 labels including 1001, 1032, 1033, 2001, 2032,
  2033, 72, 77 and 80; `tissues_info.csv`; ITK-SNAP label file) as package data, and `checksums.sha256` for the
  Harvard Dataverse archives and the cleaned labels.
- Documentation: Mindboggle-101 quickstart, tuning and checking the registration (including the NiftyReg warnings).
- pytest suite on synthetic data; tests that need NiftyReg/NiftySeg are marked `integration`.
- ruff, pre-commit, GitHub Actions (lint, tests on Python 3.9-3.12, build), `CONTRIBUTING.md`.
- `tests/regression`: a fixed small multi-atlas run that records hashes of the results, to check that refactoring
  does not change the output.

### Changed
- `src.utils.definitions` is now a thin, lazily evaluated view of the `GifConfig` defaults; assigning to its names
  still works for old scripts. Nothing is executed at import time (no multiprocessing import, no change of nibabel
  settings).
- NiftyReg and NiftySeg are run with `subprocess.run(..., check=True)` without a shell. A failing tool now stops the
  run with an error that contains the command and the end of its output (before, failures of `reg_aladin`, `reg_f3d`
  and `seg_EM` were ignored). The output of the tools is logged at DEBUG level; a successful call that printed
  NiftyReg warnings logs one INFO line.
- Library code uses `logging` instead of `print`, exceptions instead of `assert` for input validation, and `pathlib`.
  Scripts that relied on the printed progress configure logging themselves.
- Python 3.9 or newer is required (the documentation said 3.8, which was not tested).
- The SLURM scripts moved to `examples/slurm/`; the Mindboggle42 scripts moved to `scripts/mindboggle42/`;
  `run_registration_parcellation_basic.py`, `run_registration_parcellation_mb42.py`, `run_evaluation_mb42.py` and
  `clean_small_labels.py` remain as thin wrappers. They no longer contain hardcoded paths and take arguments.

### Fixed
- `get_structure_seg_from_tissue_seg` failed with an `OverflowError` on NumPy 2.
- With `MULTIPROCESSING = False` the loading of the warped atlases still used a multiprocessing pool that was not imported.
- Temporary files in the atlas folders are removed without a shell (paths with spaces).

### Removed
- `setup.py` (replaced by `pyproject.toml`), the data-dependent `tests/test_multi_atlas_segmentation*.py` scripts
  (replaced by `examples/run_multi_atlas_segmentation.py`), and all `itksnap` launching from library code and batch scripts
  (`gif-python segment --view` is the only place).

### Not changed
- The algorithm and all default parameter values. Several defaults differ from the C++ GIF (for example the MRF weight of
  `seg_EM`: 0.1 here and 0.15 in the C++ constructor); this is documented in `GifConfig`, not changed.
