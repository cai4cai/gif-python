# Contributing

## Set-up

```bash
git clone https://github.com/cai4cai/gif-python.git && cd gif-python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

NiftyReg and NiftySeg are only needed for the tests marked `integration` and for real runs (see [INSTALL.md](INSTALL.md)).

## Checks

```bash
pytest -q                      # unit tests on synthetic data, no datasets needed
pytest -q -m integration       # end-to-end test; needs reg_aladin, reg_f3d, reg_resample and seg_EM
ruff check . && ruff format --check .
python -m build                # sdist and wheel
```

The same checks run in GitHub Actions on Python 3.9 to 3.12. Tests must not need data on disk: create small
volumes in `tmp_path` (see `tests/conftest.py`). Tests that start NiftyReg or NiftySeg get `@pytest.mark.integration`;
they are skipped automatically when the executables are not found.

## Conventions

- The distribution is `pygif` and the import path is `src`; both are kept for compatibility.
- Parameters of the algorithm belong into `GifConfig` (`src/utils/config.py`), not into module globals. A new
  parameter needs a default that keeps the previous behaviour. Changing an existing default changes results and needs
  a discussion in an issue first.
- No work at import time and no hardcoded paths. Library code logs with `logging`, validates with exceptions, and
  runs external tools through `src.utils.process.run_tool` (no shell).
- Keep Python 3.9 compatibility (`from __future__ import annotations`; `typing.Optional` in dataclass fields).
- One logical change per commit with an imperative subject line; formatting-only changes in their own commit.
- Update `CHANGELOG.md` for user-visible changes.

## Checking that a refactoring does not change the results

`tests/regression/README.md` describes a small fixed multi-atlas run that records SHA-256 hashes of the parcellation,
the tissue segmentation and the fusion weights. Run it before and after a change that must not alter the output.
