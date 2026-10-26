"""Small helpers around nibabel."""

from __future__ import annotations

import nibabel as nib
import numpy as np


def relax_quaternion_check() -> None:
    """Accept NIfTI headers whose qform quaternion is slightly non-unit.

    Some datasets (e.g. CANDI) do not pass nibabel's default consistency check. The setting is a
    process-wide class attribute of nibabel, so it is applied explicitly where images are read
    instead of as a side effect of importing this package. Calling it repeatedly is harmless.
    """
    nib.Nifti1Header.quaternion_threshold = -np.finfo(np.float32).eps * 10
