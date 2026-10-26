"""Generate ``structures_info.csv`` and ``tissues_info.csv`` from an atlas segmentation.

The labels are taken from one or more segmentation volumes, their names and colours from a
FreeSurfer colour table (``FreeSurferColorLUT.txt`` format: ``index name R G B A`` per line), and
the tissue classes of each structure are assigned by name patterns (:data:`DEFAULT_TISSUE_RULES`,
which follow the FreeSurfer naming conventions and reproduce the Mindboggle-101 table shipped in
``src/resources/mindboggle101``). Labels whose name matches no rule are reported and, unless
``strict`` is set, may belong to any tissue.
"""

from __future__ import annotations

import colorsys
import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

#: tissue classes of ``tissues_info.csv`` (same as in the Mindboggle-101 resources and INSTALL.md)
DEFAULT_TISSUES: Tuple[Tuple[int, str], ...] = (
    (0, "Non-Brain Outer Tissue"),
    (1, "Cerebral Spinal Fluid"),
    (2, "Grey Matter"),
    (3, "White Matter"),
    (4, "Deep Grey Matter"),
    (5, "Brain Stem and Pons"),
)

#: (regular expression, tissues); the first rule that matches the normalised name (lower case,
#: spaces replaced by '-') is used.
DEFAULT_TISSUE_RULES: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    (r"^(unknown|background|unlabeled|unlabelled)$", (0,)),
    (r"^ctx[-_](lh|rh)[-_]unknown$|^ctx[-_]unknown$", (0, 1, 2, 3, 4, 5)),
    (r"^ctx[-_]", (2,)),
    (r"^non-wm-hypointensities$", (2, 4, 5)),
    (r"^wm-hypointensities$", (3,)),
    (r"cerebral-white-matter|cerebellum-white-matter|^cc[-_]", (3,)),
    (r"cerebellum-cortex|cerebral-cortex", (2,)),
    (r"hippocampus|amygdala", (2, 4)),
    (r"ventraldc", (3, 4, 5)),
    (r"thalamus|caudate|putamen|pallidum|accumbens", (3, 4)),
    (r"brain-?stem", (5,)),
    (r"choroid-plexus", (1, 2, 4, 5)),
    (r"vessel|optic-chiasm", (4,)),
    (r"ventricle|inf-lat-vent|^csf$", (1,)),
)


@dataclass(frozen=True)
class ColorEntry:
    """One line of a FreeSurfer colour table."""

    name: str
    rgba: Tuple[int, int, int, int]


def read_color_table(path: PathLike) -> Dict[int, ColorEntry]:
    """Parse a FreeSurfer colour table (``FreeSurferColorLUT.txt`` format).

    Comment lines (``#``) and blank lines are ignored.

    :raises ValueError: for a line that is not ``index name R G B A``
    """
    table: Dict[int, ColorEntry] = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            if len(parts) < 6:
                raise ValueError
            index = int(parts[0])
            rgba = tuple(int(v) for v in parts[-4:])
        except ValueError:
            raise ValueError(f"{path}:{number}: expected 'index name R G B A', got {line!r}") from None
        # FreeSurfer names contain no spaces; tolerate them anyway (everything between index and colour)
        table[index] = ColorEntry(" ".join(parts[1:-4]), rgba)  # type: ignore[arg-type]
    return table


def labels_in_segmentations(paths: Iterable[PathLike]) -> List[int]:
    """Sorted union of the label values in the given NIfTI segmentations.

    :raises ValueError: if a volume contains non-integer or negative values
    """
    labels: set = set()
    for path in paths:
        data = np.asanyarray(nib.load(str(path)).dataobj)
        if not np.issubdtype(data.dtype, np.integer):
            if not np.array_equal(np.rint(data), data):
                raise ValueError(f"{path} contains non-integer values; it is not a label map")
        values = np.unique(data).astype(np.int64)
        if values.min() < 0:
            raise ValueError(f"{path} contains negative labels")
        labels.update(int(v) for v in values)
    return sorted(labels)


def normalise_name(name: str) -> str:
    return name.strip().lower().replace(" ", "-")


def read_tissue_rules(path: PathLike) -> List[Tuple[str, Tuple[int, ...]]]:
    """Read extra rules from a csv with the columns ``pattern`` (regular expression) and ``tissues``
    (for example ``[2]`` or ``"[3, 4]"``)."""
    from ast import literal_eval

    rules = []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            tissues = literal_eval(row["tissues"])
            rules.append((row["pattern"], tuple(int(t) for t in tissues)))
    return rules


def assign_tissues(
    name: str,
    rules: Sequence[Tuple[str, Sequence[int]]] = DEFAULT_TISSUE_RULES,
) -> Optional[Tuple[int, ...]]:
    """Tissues of a structure from its name, or ``None`` if no rule matches."""
    normalised = normalise_name(name)
    for pattern, tissues in rules:
        if re.search(pattern, normalised):
            return tuple(tissues)
    return None


def build_structures(
    labels: Sequence[int],
    color_table: Optional[Dict[int, ColorEntry]] = None,
    rules: Sequence[Tuple[str, Sequence[int]]] = DEFAULT_TISSUE_RULES,
    tissue_ids: Sequence[int] = tuple(t for t, _ in DEFAULT_TISSUES),
    strict: bool = False,
) -> Tuple[List[Tuple[int, str, Tuple[int, ...]]], List[int], List[int]]:
    """Name and tissue assignment for each label.

    :return: ``(rows (label, name, tissues), labels without a name, labels without a tissue rule)``
    :raises ValueError: with ``strict`` if a label has no name in the colour table or no tissue rule
    """
    color_table = color_table or {}
    rows, unnamed, unassigned = [], [], []
    for label in labels:
        if label in color_table:
            name = color_table[label].name
        elif label == 0:
            name = "Unknown"
        else:
            name = f"label-{label}"
            unnamed.append(label)
        tissues = assign_tissues(name, rules)
        if tissues is None:
            unassigned.append(label)
            tissues = tuple(tissue_ids)
        rows.append((label, name, tissues))
    if strict and (unnamed or unassigned):
        raise ValueError(
            f"strict mode: labels without a name in the colour table: {unnamed}; "
            f"labels whose name matches no tissue rule: {unassigned}"
        )
    unknown = {t for _, _, tissues in rows for t in tissues} - set(tissue_ids)
    if unknown:
        raise ValueError(f"The tissue rules use tissue numbers {sorted(unknown)} that are not in the tissue table")
    return rows, unnamed, unassigned


def write_structures_csv(rows: Sequence[Tuple[int, str, Sequence[int]]], path: PathLike) -> None:
    """Write ``label,name,tissues`` (tissues as a Python list, e.g. ``"[3, 4]"``)."""
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["label", "name", "tissues"])
        for label, name, tissues in rows:
            writer.writerow([label, name, str(list(tissues))])


def write_tissues_csv(tissues: Sequence[Tuple[int, str]], path: PathLike) -> None:
    """Write ``label,name`` of the tissue classes."""
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["label", "name"])
        writer.writerows(tissues)


def read_tissues_csv(path: PathLike) -> List[Tuple[int, str]]:
    with Path(path).open(newline="") as f:
        return [(int(row["label"]), row["name"]) for row in csv.DictReader(f)]


def _generated_color(label: int) -> Tuple[int, int, int]:
    """Deterministic, well separated colour for labels without an entry in the colour table."""
    hue = (label * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def write_itksnap_labels(
    rows: Sequence[Tuple[int, str, Sequence[int]]],
    path: PathLike,
    color_table: Optional[Dict[int, ColorEntry]] = None,
) -> None:
    """Write an ITK-SNAP label description file (colours from the colour table where available)."""
    color_table = color_table or {}
    lines = [
        "################################################",
        "# ITK-SNAP Label Description File",
        "# File format:",
        "# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL",
        "################################################",
    ]
    for label, name, _ in rows:
        r, g, b = color_table[label].rgba[:3] if label in color_table else _generated_color(label)
        visible = 0 if label == 0 else 1
        lines.append(f'{label:5d} {r:4d} {g:4d} {b:4d} {visible:9d} {visible:2d} {visible:2d}    "{name}"')
    Path(path).write_text("\n".join(lines) + "\n")


def make_structure_info(
    segmentations: Sequence[PathLike],
    out_dir: PathLike,
    color_table_path: Optional[PathLike] = None,
    tissue_rules_path: Optional[PathLike] = None,
    tissues_info_path: Optional[PathLike] = None,
    strict: bool = False,
    write_itksnap: bool = True,
) -> Dict[str, Path]:
    """Create ``structures_info.csv`` and ``tissues_info.csv`` (and an ITK-SNAP label file).

    :param segmentations: atlas segmentation volumes; every label found in any of them is listed
    :param out_dir: output folder (created if needed)
    :param color_table_path: FreeSurfer colour table with the names (and colours) of the labels
    :param tissue_rules_path: csv with extra ``pattern,tissues`` rules, applied before the default rules
    :param tissues_info_path: existing ``tissues_info.csv`` to use instead of the default six tissues
    :param strict: fail instead of guessing for labels without name or tissue rule
    :return: the written files by name
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = labels_in_segmentations(segmentations)
    color_table = read_color_table(color_table_path) if color_table_path else None
    rules = list(read_tissue_rules(tissue_rules_path)) if tissue_rules_path else []
    rules += list(DEFAULT_TISSUE_RULES)
    tissues = read_tissues_csv(tissues_info_path) if tissues_info_path else list(DEFAULT_TISSUES)

    rows, unnamed, unassigned = build_structures(labels, color_table, rules, [t for t, _ in tissues], strict)
    if unnamed:
        logger.warning(
            "%d label(s) are not in the colour table and are named 'label-N': %s%s",
            len(unnamed),
            unnamed,
            "" if color_table else " (no --color-table given)",
        )
    if unassigned:
        logger.warning(
            "%d label(s) match no tissue rule and were assigned to all tissues; edit the csv or pass --tissue-rules: %s",
            len(unassigned),
            unassigned,
        )

    written = {
        "structures_info.csv": out_dir / "structures_info.csv",
        "tissues_info.csv": out_dir / "tissues_info.csv",
    }
    write_structures_csv(rows, written["structures_info.csv"])
    write_tissues_csv(tissues, written["tissues_info.csv"])
    if write_itksnap:
        written["itksnap_label_descriptions.txt"] = out_dir / "itksnap_label_descriptions.txt"
        write_itksnap_labels(rows, written["itksnap_label_descriptions.txt"], color_table)
    return written
