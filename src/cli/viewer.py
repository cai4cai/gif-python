"""Optional launching of ITK-SNAP (only used by command line flags such as ``--view``)."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def open_in_itksnap(image: PathLike, segmentation: PathLike, label_file: Optional[PathLike] = None) -> bool:
    """Open ``image`` with ``segmentation`` as overlay in ITK-SNAP.

    :return: ``False`` (with a warning) if ITK-SNAP is not installed or fails to start.
    """
    executable = shutil.which("itksnap")
    if executable is None:
        logger.warning("--view: the 'itksnap' executable was not found on PATH; open %s manually", segmentation)
        return False
    command = [executable, "-g", str(image), "-s", str(segmentation)]
    if label_file:
        command += ["-l", str(label_file)]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("--view: could not run ITK-SNAP: %s", exc)
        return False
    return True
