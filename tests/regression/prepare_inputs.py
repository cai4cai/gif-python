"""Build structures_info.csv / tissues_info.csv from a GIF-style ``labels.xml``.

This is only used to set up the regression case described in ``README.md``. The
XML is the one written by the C++ GIF database builder: a ``<labels>`` list with
``<number>`` and ``<tissues>`` per label and a ``<tissues>`` list with the tissue
names. Tissue numbers are re-numbered to 0..N-1 (in increasing order of the
original number) because the Python pipeline allocates one prior channel per
tissue number.

Usage::

    python prepare_inputs.py --labels-xml labels.xml --label-map label_map.tsv --out-dir inputs/
"""

from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-xml", required=True, type=Path)
    ap.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="optional TSV with columns original, gif, tissue; used only to build structure names",
    )
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    root = ET.parse(args.labels_xml).getroot()
    label_tissue = {
        int(i.findtext("number")): [int(t) for t in i.findtext("tissues").split(",")]
        for i in root.find("labels").findall("item")
    }
    tissue_names = {
        int(i.findtext("number")): i.findtext("name") for i in root.find("tissues").findall("item")
    }
    renumber = {old: new for new, old in enumerate(sorted(tissue_names))}

    names = {}
    if args.label_map is not None:
        with args.label_map.open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                names[int(row["gif"])] = f"label-{row['original']}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "structures_info.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "name", "tissues"])
        w.writerow([0, "Background", "[0]"])
        for label in sorted(label_tissue):
            tissues = sorted(renumber[t] for t in label_tissue[label])
            w.writerow([label, names.get(label, f"label-{label}"), str(tissues)])
    with (args.out_dir / "tissues_info.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "name"])
        for old in sorted(tissue_names):
            w.writerow([renumber[old], tissue_names[old]])
    print(f"wrote {len(label_tissue) + 1} structures and {len(tissue_names)} tissues to {args.out_dir}")


if __name__ == "__main__":
    main()
