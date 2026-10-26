"""Replace sparse labels of a segmentation by the most frequent label among their neighbours.

This is the vectorised version of the per-voxel loop of ``clean_small_labels.py``. For every voxel
that carries one of the labels to replace:

* the 26 neighbours (3x3x3 neighbourhood without the voxel itself) are collected from the
  *original* segmentation, and neighbours with the same label as the voxel are ignored;
* the voxel receives the most frequent remaining label; among labels with the same count the
  smallest label wins (``np.unique`` + ``np.argmax`` in the loop);
* a voxel without any neighbour of another label raises an error, and so does a result that is
  itself one of the labels to replace.

Neighbours outside the volume are ignored. (The original loop wrapped around at the low borders and
failed with an ``IndexError`` at the high borders; sparse labels do not occur there in practice.)
"""

from __future__ import annotations

import glob
import itertools
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

#: The six cortical labels that are present in only a few Mindboggle-101 subjects.
DEFAULT_SPARSE_LABELS: Tuple[int, ...] = (1001, 1032, 1033, 2001, 2032, 2033)

#: Opt-in labels (``--labels``): 5th ventricle, non-WM hypointensities and WM hypointensities.
OPTIONAL_LABELS: Tuple[int, ...] = (72, 80, 77)

#: Offsets of the 26 neighbours.
NEIGHBOUR_OFFSETS = np.array([o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)], dtype=np.int64)

# marker for "no neighbour": outside the volume, or a neighbour carrying the label itself
_INVALID = np.iinfo(np.int64).max

#: voxels processed at once (memory is about chunk * 26 * 26 bytes)
_CHUNK = 20000


def _as_integer_labels(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.int64, copy=False)
    rounded = np.rint(data)
    if not np.array_equal(rounded, data):
        raise ValueError("The segmentation contains non-integer values; it is not a label map")
    return rounded.astype(np.int64)


def _most_frequent_neighbour(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mode of each row (``_INVALID`` entries do not count); smallest value wins ties.

    :return: ``(mode, number of valid neighbours)`` per row.
    """
    values = np.sort(values, axis=1)  # ascending: invalid entries are last
    valid = values != _INVALID
    counts = (values[:, :, None] == values[:, None, :]).sum(axis=2) * valid
    best = counts.argmax(axis=1)  # first maximum = smallest label among equal counts
    return values[np.arange(len(values)), best], valid.sum(axis=1)


def clean_labels_array(
    data: np.ndarray, labels_to_replace: Iterable[int] = DEFAULT_SPARSE_LABELS
) -> Tuple[np.ndarray, int]:
    """Replace the voxels with one of ``labels_to_replace`` by their most frequent neighbouring label.

    :param data: 3D label volume (integer values, possibly stored as float)
    :param labels_to_replace: labels to remove
    :return: ``(cleaned volume with the dtype of data, number of changed voxels)``
    :raises ValueError: if a voxel has no neighbour with another label, or the replacement is one of
        the labels to replace, or ``data`` is not a 3D label map
    """
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got an array with shape {data.shape}")
    labels_to_replace = [int(label) for label in labels_to_replace]
    labels = _as_integer_labels(data)
    padded = np.pad(labels, 1, mode="constant", constant_values=_INVALID)
    out = labels.copy()

    for label in labels_to_replace:
        coords = np.argwhere(labels == label)
        for start in range(0, len(coords), _CHUNK):
            block = coords[start : start + _CHUNK]
            # neighbour values of all voxels in the block: (n, 26); coords + 1 because of the padding
            neighbours = padded[
                block[:, None, 0] + 1 + NEIGHBOUR_OFFSETS[None, :, 0],
                block[:, None, 1] + 1 + NEIGHBOUR_OFFSETS[None, :, 1],
                block[:, None, 2] + 1 + NEIGHBOUR_OFFSETS[None, :, 2],
            ]
            neighbours = np.where(neighbours == label, _INVALID, neighbours)
            mode, n_valid = _most_frequent_neighbour(neighbours)

            isolated = np.flatnonzero(n_valid == 0)
            if isolated.size:
                voxel = tuple(int(v) for v in block[isolated[0]])
                raise ValueError(f"Voxel {voxel} with label {label} has no neighbour with a different label")
            forbidden = np.flatnonzero(np.isin(mode, labels_to_replace))
            if forbidden.size:
                voxel = tuple(int(v) for v in block[forbidden[0]])
                raise ValueError(
                    f"The most frequent neighbouring label of voxel {voxel} (label {label}) is "
                    f"{int(mode[forbidden[0]])}, which is also in the labels to replace {labels_to_replace}"
                )
            out[tuple(block.T)] = mode

    n_changed = int(np.count_nonzero(out != labels))
    return out.astype(data.dtype), n_changed


def output_path(path: PathLike, suffix: str = "_cleaned") -> Path:
    """``a/labels.nii.gz`` -> ``a/labels_cleaned.nii.gz`` (also for ``.nii``)."""
    path = Path(path)
    name = path.name
    for extension in (".nii.gz", ".nii"):
        if name.endswith(extension):
            return path.with_name(name[: -len(extension)] + suffix + extension)
    raise ValueError(f"Not a NIfTI file name (.nii or .nii.gz): {path}")


def clean_labels_file(
    path_in: PathLike,
    path_out: Optional[PathLike] = None,
    labels_to_replace: Iterable[int] = DEFAULT_SPARSE_LABELS,
    suffix: str = "_cleaned",
) -> Tuple[Path, int]:
    """Clean one segmentation file; the output keeps the data type and header of the input.

    :param path_out: output file; default ``path_in`` with ``suffix`` appended to the name
    :return: ``(output path, number of changed voxels)``
    """
    path_in = Path(path_in)
    path_out = Path(path_out) if path_out is not None else output_path(path_in, suffix)
    nii = nib.load(str(path_in))
    data = np.asanyarray(nii.dataobj)
    cleaned, n_changed = clean_labels_array(data, labels_to_replace)
    header = nii.header.copy()
    header.set_data_dtype(cleaned.dtype)
    nib.save(nib.Nifti1Image(cleaned, nii.affine, header), str(path_out))
    return path_out, n_changed


def expand_inputs(patterns: Sequence[str], suffix: str = "_cleaned") -> List[Path]:
    """Expand glob patterns and file names; files that already carry ``suffix`` are skipped.

    :raises FileNotFoundError: if nothing matches
    """
    files: List[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        files.extend(Path(m) for m in matches)
    files = [f for f in dict.fromkeys(files) if not _is_output_name(f, suffix)]
    if not files:
        raise FileNotFoundError(f"No input file matches {' '.join(patterns)}")
    return files


def _is_output_name(path: Path, suffix: str) -> bool:
    return any(path.name.endswith(suffix + extension) for extension in (".nii.gz", ".nii"))
