"""Data files shipped with the package."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

MINDBOGGLE101 = "src.resources.mindboggle101"


def resource_path(package: str, name: str) -> Path:
    """Path of the data file ``name`` of ``package`` (the package is installed unzipped)."""
    return Path(str(resources.files(package).joinpath(name)))


def mindboggle101_file(name: str) -> Path:
    """Path of one of the Mindboggle-101 files (see ``src/resources/mindboggle101/README.md``)."""
    path = resource_path(MINDBOGGLE101, name)
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not part of the installed Mindboggle-101 resources")
    return path
