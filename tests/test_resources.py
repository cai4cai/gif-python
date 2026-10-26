import re
from ast import literal_eval

import pandas as pd
import pytest

from src.resources import mindboggle101_file

SIX_SPARSE = [1001, 1032, 1033, 2001, 2032, 2033]


@pytest.fixture(scope="module")
def structures():
    return pd.read_csv(mindboggle101_file("structures_info.csv"), index_col="label", converters={"tissues": literal_eval})


@pytest.fixture(scope="module")
def tissues():
    return pd.read_csv(mindboggle101_file("tissues_info.csv"), index_col="label")


def test_all_labels_of_issue_1_are_present_as_grey_matter(structures):
    for label in SIX_SPARSE:
        assert structures.loc[label, "tissues"] == [2], label
        assert structures.loc[label, "name"].startswith("ctx-")


def test_problematic_labels_are_listed(structures):
    assert structures.loc[72, "tissues"] == [1]
    assert structures.loc[77, "tissues"] == [3]
    assert structures.loc[80, "tissues"] == [2, 4, 5]
    assert {0, 24, 85, 1000, 2000} <= set(structures.index)


def test_dkt31_cortical_labels_complete(structures):
    expected = {h * 1000 + i for h in (1, 2) for i in list(range(1, 36)) if i != 4}  # 1004/2004 corpus callosum: not in DKT31
    assert expected <= set(structures.index)


def test_tissue_numbers_are_defined(structures, tissues):
    used = {t for ts in structures["tissues"] for t in ts}
    assert used <= set(tissues.index)
    assert list(tissues.index) == list(range(6))


def test_labels_are_unique_and_sorted(structures):
    assert structures.index.is_unique and structures.index.is_monotonic_increasing
    assert structures["name"].is_unique


def test_itksnap_file_matches_the_csv(structures):
    entries = {}
    for line in mindboggle101_file("itksnap_label_descriptions_mindboggle101.txt").read_text().splitlines():
        match = re.match(r'\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\d)\s+(\d)\s+"(.*)"\s*$', line)
        if match:
            entries[int(match.group(1))] = (tuple(int(match.group(i)) for i in (2, 3, 4)), match.group(8))
    assert set(entries) == set(structures.index)
    assert all(entries[label][1] == structures.loc[label, "name"] for label in entries)
    for label in SIX_SPARSE:
        assert label in entries
    colours = [colour for colour, _ in entries.values()]
    for label in (1001, 1032, 1033, 2001):  # added with unused colours
        assert colours.count(entries[label][0]) == 1


def test_missing_resource_is_reported():
    with pytest.raises(FileNotFoundError):
        mindboggle101_file("nope.csv")


def test_csv_is_usable_by_the_pipeline_reader(structures):
    # the pipeline reads the file with index_col="label" and a literal_eval converter
    assert all(isinstance(ts, list) and ts for ts in structures["tissues"])
