import csv
import logging
from ast import literal_eval

import numpy as np
import pandas as pd
import pytest

from src.cli.main import main
from src.labels.structure_info import (
    DEFAULT_TISSUE_RULES,
    assign_tissues,
    build_structures,
    labels_in_segmentations,
    make_structure_info,
    read_color_table,
)
from src.resources import mindboggle101_file


@pytest.fixture(scope="module")
def shipped():
    return pd.read_csv(mindboggle101_file("structures_info.csv"), converters={"tissues": literal_eval})


def write_lut(path, entries):
    lines = ["#No. Label Name:                            R   G   B   A"]
    lines += [f"{label:<4d} {name:<40s} {r:3d} {g:3d} {b:3d} 0" for label, name, (r, g, b) in entries]
    path.write_text("\n".join(lines) + "\n")


def test_rules_reproduce_the_shipped_mindboggle_table(shipped):
    mismatches = [
        (row.label, row.name, row.tissues, assign_tissues(row.name))
        for row in shipped.itertuples()
        if assign_tissues(row.name) != tuple(row.tissues)
    ]
    assert mismatches == []


def test_end_to_end_reproduces_the_shipped_table(tmp_path, write_nifti, shipped):
    labels = shipped["label"].to_numpy()
    volume = np.zeros((len(labels), 1, 1), dtype=np.int32)
    volume[:, 0, 0] = labels
    seg = write_nifti(volume, tmp_path / "seg.nii.gz")
    lut = tmp_path / "LUT.txt"
    write_lut(lut, [(r.label, r.name, (r.label % 256, 10, 20)) for r in shipped.itertuples() if r.label != 0])
    out = make_structure_info([seg], tmp_path / "out", color_table_path=lut, strict=True)
    generated = pd.read_csv(out["structures_info.csv"], converters={"tissues": literal_eval})
    pd.testing.assert_frame_equal(generated, shipped)
    assert (out["tissues_info.csv"].read_text() == mindboggle101_file("tissues_info.csv").read_text())


def test_read_color_table_skips_comments_and_validates(tmp_path):
    lut = tmp_path / "lut.txt"
    lut.write_text("# comment\n\n0   Unknown  0 0 0 0\n17  Left-Hippocampus  220 216 20 0\n")
    table = read_color_table(lut)
    assert table[17].name == "Left-Hippocampus" and table[17].rgba == (220, 216, 20, 0)
    lut.write_text("17 Left-Hippocampus 220 216\n")
    with pytest.raises(ValueError, match="lut.txt:1"):
        read_color_table(lut)


def test_labels_are_the_union_over_all_segmentations(tmp_path, write_nifti):
    a = write_nifti(np.array([[[0, 3]]], dtype=np.uint8), tmp_path / "a.nii.gz")
    b = write_nifti(np.array([[[3, 9]]], dtype=np.int16), tmp_path / "b.nii.gz")
    assert labels_in_segmentations([a, b]) == [0, 3, 9]
    f = write_nifti(np.array([[[0.5]]], dtype=np.float32), tmp_path / "f.nii.gz")
    with pytest.raises(ValueError, match="non-integer"):
        labels_in_segmentations([f])


def test_float_label_maps_with_integer_values_are_accepted(tmp_path, write_nifti):
    f = write_nifti(np.array([[[0.0, 2.0, 2.0]]], dtype=np.float32), tmp_path / "f.nii.gz")
    assert labels_in_segmentations([f]) == [0, 2]


def test_unnamed_and_unmatched_labels(caplog):
    table = read_color_table_from_text("5  Some-Odd-Structure  1 2 3 0")
    rows, unnamed, unassigned = build_structures([0, 5, 6], table)
    assert rows[0] == (0, "Unknown", (0,))
    assert rows[1] == (5, "Some-Odd-Structure", (0, 1, 2, 3, 4, 5))
    assert rows[2][1] == "label-6"
    assert unnamed == [6] and unassigned == [5, 6]
    with pytest.raises(ValueError, match="strict"):
        build_structures([0, 5], table, strict=True)


def read_color_table_from_text(text):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "lut.txt"
        p.write_text(text + "\n")
        return read_color_table(p)


def test_warnings_for_guessed_assignments(tmp_path, write_nifti, caplog):
    seg = write_nifti(np.array([[[0, 7]]], dtype=np.uint8), tmp_path / "seg.nii.gz")
    with caplog.at_level(logging.WARNING):
        make_structure_info([seg], tmp_path / "o")
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "[7]" in messages and "no tissue rule" in messages


def test_custom_rules_take_precedence(tmp_path, write_nifti):
    seg = write_nifti(np.array([[[0, 7]]], dtype=np.uint8), tmp_path / "seg.nii.gz")
    lut = tmp_path / "lut.txt"
    write_lut(lut, [(7, "Left-Hippocampus", (1, 2, 3))])
    rules = tmp_path / "rules.csv"
    rules.write_text('pattern,tissues\nhippocampus,"[3, 4]"\n')
    out = make_structure_info([seg], tmp_path / "o", color_table_path=lut, tissue_rules_path=rules)
    rows = list(csv.DictReader(out["structures_info.csv"].open()))
    assert rows[1]["tissues"] == "[3, 4]"


def test_itksnap_file_uses_the_colour_table(tmp_path, write_nifti):
    seg = write_nifti(np.array([[[0, 17, 9999]]], dtype=np.uint16), tmp_path / "seg.nii.gz")
    lut = tmp_path / "lut.txt"
    write_lut(lut, [(17, "Left-Hippocampus", (220, 216, 20))])
    out = make_structure_info([seg], tmp_path / "o", color_table_path=lut)
    text = out["itksnap_label_descriptions.txt"].read_text()
    assert '17  220  216   20' in text and '"Left-Hippocampus"' in text and '"label-9999"' in text
    assert make_structure_info([seg], tmp_path / "o2", write_itksnap=False).get("itksnap_label_descriptions.txt") is None


def test_custom_tissue_table_is_validated(tmp_path, write_nifti):
    seg = write_nifti(np.array([[[0, 7]]], dtype=np.uint8), tmp_path / "seg.nii.gz")
    tissues = tmp_path / "tissues.csv"
    tissues.write_text("label,name\n0,Other\n1,Brain\n")
    lut = tmp_path / "lut.txt"
    write_lut(lut, [(7, "Left-Hippocampus", (1, 2, 3))])  # rule assigns tissues 2 and 4
    with pytest.raises(ValueError, match="not in the tissue table"):
        make_structure_info([seg], tmp_path / "o", color_table_path=lut, tissues_info_path=tissues)


def test_rules_are_ordered_specific_before_general():
    assert assign_tissues("non-WM-hypointensities") == (2, 4, 5)
    assert assign_tissues("WM-hypointensities") == (3,)
    assert assign_tissues("Left-Cerebellum-White-Matter") == (3,)
    assert assign_tissues("Right-Cerebellum-Cortex") == (2,)
    assert assign_tissues("ctx-rh-unknown") == (0, 1, 2, 3, 4, 5)
    assert assign_tissues("CSF") == (1,)
    assert assign_tissues("Something-else") is None
    assert len(DEFAULT_TISSUE_RULES) > 5


def test_cli(tmp_path, write_nifti, capsys):
    write_nifti(np.array([[[0, 17, 18]]], dtype=np.uint8), tmp_path / "seg_1.nii.gz")
    lut = tmp_path / "lut.txt"
    write_lut(lut, [(17, "Left-Hippocampus", (1, 2, 3)), (18, "Left-Amygdala", (4, 5, 6))])
    code = main(["make-structure-info", str(tmp_path / "seg_*.nii.gz"), "--color-table", str(lut), "--out-dir", str(tmp_path / "o")])
    assert code == 0
    df = pd.read_csv(tmp_path / "o" / "structures_info.csv", index_col="label", converters={"tissues": literal_eval})
    assert df.loc[17, "tissues"] == [2, 4] and df.loc[0, "name"] == "Unknown"
    assert main(["make-structure-info", str(tmp_path / "none*.nii.gz"), "--out-dir", str(tmp_path / "o")]) == 1
