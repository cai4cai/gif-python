import nibabel as nib
import numpy as np
import pytest

from src.cli.main import main
from src.labels.cleaning import (
    DEFAULT_SPARSE_LABELS,
    OPTIONAL_LABELS,
    clean_labels_array,
    clean_labels_file,
    expand_inputs,
    output_path,
)

SPARSE = list(DEFAULT_SPARSE_LABELS)


def reference_loop(data, labels_to_replace):
    """The per-voxel loop of the original clean_small_labels.py (printing removed)."""
    offsets = np.array(
        [
            [-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
            [-1, 1, -1], [-1, 1, 0], [-1, 1, 1], [0, -1, -1], [0, -1, 0], [0, -1, 1],
            [0, 0, -1], [0, 0, 1], [0, 1, -1], [0, 1, 0], [0, 1, 1],
            [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1],
            [1, 1, -1], [1, 1, 0], [1, 1, 1],
        ]
    )  # fmt: skip
    data = data.astype(np.float64)
    data_out = np.copy(data)
    for l_in in labels_to_replace:
        locs = np.where(data == l_in)
        for i, j, k in zip(*locs):
            neighbor_vals = []
            for n in np.array([i, j, k]) + offsets:
                neighbor_val = data[n[0], n[1], n[2]]
                if neighbor_val != l_in:
                    neighbor_vals.append(neighbor_val)
            if len(neighbor_vals) == 0:
                raise Exception("no neighbors found...")
            labels, counts = np.unique(np.array(neighbor_vals), return_counts=True)
            max_label = labels[np.argmax(counts)]
            assert max_label not in labels_to_replace, f"{max_label=} in {labels_to_replace=}"
            data_out[i, j, k] = max_label
    return data_out


def synthetic_volume(seed, shape=(22, 22, 22), background=(0, 2, 41, 1002, 1003, 2002), blobs=12):
    """Random labels with sparse blobs (at most 2x2x2) that are separated from each other and from the border."""
    rng = np.random.default_rng(seed)
    data = rng.choice(background, size=shape).astype(np.int32)
    cells = [(x, y, z) for x in (2, 7, 12, 17) for y in (2, 7, 12, 17) for z in (2, 7, 12, 17)]
    for index in rng.choice(len(cells), size=blobs, replace=False):
        label = int(rng.choice(SPARSE))
        size = rng.integers(1, 3, size=3)
        data[tuple(slice(s, s + n) for s, n in zip(cells[index], size))] = label
    return data


@pytest.mark.parametrize("seed", range(8))
def test_equivalent_to_the_original_per_voxel_loop(seed):
    data = synthetic_volume(seed)
    assert np.isin(data, SPARSE).sum() > 0
    expected = reference_loop(data, SPARSE)
    cleaned, n_changed = clean_labels_array(data, SPARSE)
    np.testing.assert_array_equal(cleaned, expected.astype(data.dtype))
    assert n_changed == int((expected != data).sum()) > 0
    assert not np.isin(cleaned, SPARSE).any()


@pytest.mark.parametrize("seed", range(3))
def test_equivalent_with_all_nine_labels(seed):
    # 72/80/77 are replaced as well; background labels do not contain them
    labels = SPARSE + list(OPTIONAL_LABELS)
    rng = np.random.default_rng(100 + seed)
    data = rng.choice([0, 2, 41, 1002], size=(12, 12, 12)).astype(np.int16)
    for label in labels:
        i, j, k = rng.integers(2, 9, size=3)
        data[i, j, k] = label
    np.testing.assert_array_equal(clean_labels_array(data, labels)[0], reference_loop(data, labels).astype(np.int16))


def test_ties_go_to_the_smallest_label():
    data = np.full((5, 5, 5), 7, dtype=np.int32)
    data[2, 2, 2] = 1001
    # 13 neighbours become 3, the other 13 stay 7: a tie between 3 and 7
    data[:2] = 3
    data[2, :2] = 3
    data[2, 2, :2] = 3
    assert (data[1:4, 1:4, 1:4] == 3).sum() == 13 and (data[1:4, 1:4, 1:4] == 7).sum() == 13
    cleaned, _ = clean_labels_array(data, [1001])
    assert cleaned[2, 2, 2] == 3
    np.testing.assert_array_equal(cleaned, reference_loop(data, [1001]).astype(np.int32))
    data[data == 3] = 9  # now 7 < 9 wins
    assert clean_labels_array(data, [1001])[0][2, 2, 2] == 7


def test_most_frequent_label_wins_over_smaller_label():
    data = np.full((5, 5, 5), 50, dtype=np.int32)
    data[2, 2, 2] = 2001
    data[1, 1, 1] = 4  # a single neighbour with a smaller label
    assert clean_labels_array(data, SPARSE)[0][2, 2, 2] == 50


def test_neighbours_with_the_same_label_do_not_count():
    data = np.full((5, 5, 5), 9, dtype=np.int32)
    data[1:4, 1:4, 1:3] = 1033  # most neighbours of (2, 2, 2) are sparse, remaining ones are 9
    cleaned, _ = clean_labels_array(data, SPARSE)
    assert (cleaned == 9).all()


def test_voxel_without_other_neighbour_raises_like_the_original():
    data = np.full((9, 9, 9), 5, dtype=np.int32)
    data[2:7, 2:7, 2:7] = 1001  # the centre voxel only has neighbours with the same label
    with pytest.raises(ValueError, match="no neighbour"):
        clean_labels_array(data, [1001])
    with pytest.raises(Exception, match="no neighbors"):
        reference_loop(data, [1001])


def test_replacement_that_is_itself_sparse_raises_like_the_original():
    data = np.full((5, 5, 5), 1032, dtype=np.int32)
    data[2, 2, 2] = 1001  # all neighbours are 1032
    with pytest.raises(ValueError, match="1032"):
        clean_labels_array(data, SPARSE)
    with pytest.raises(AssertionError):
        reference_loop(data, SPARSE)


def test_labels_outside_the_list_are_untouched():
    data = synthetic_volume(3)
    data[1, 1, 1] = 72
    cleaned, _ = clean_labels_array(data, SPARSE)
    assert cleaned[1, 1, 1] == 72  # 72 is not removed by default


def test_border_voxels_use_the_neighbours_inside_the_volume():
    data = np.full((4, 4, 4), 8, dtype=np.int32)
    data[0, 0, 0] = 2033
    assert clean_labels_array(data, SPARSE)[0][0, 0, 0] == 8


def test_float_volumes_and_dtype_preservation():
    data = synthetic_volume(1).astype(np.float64)
    cleaned, _ = clean_labels_array(data, SPARSE)
    assert cleaned.dtype == np.float64
    np.testing.assert_array_equal(cleaned, reference_loop(data, SPARSE))
    with pytest.raises(ValueError, match="non-integer"):
        clean_labels_array(data + 0.5, SPARSE)
    with pytest.raises(ValueError, match="3D"):
        clean_labels_array(data[0], SPARSE)


def test_more_voxels_than_one_chunk(monkeypatch):
    import src.labels.cleaning as cleaning

    monkeypatch.setattr(cleaning, "_CHUNK", 7)
    data = synthetic_volume(5, blobs=30)
    np.testing.assert_array_equal(
        clean_labels_array(data, SPARSE)[0], reference_loop(data, SPARSE).astype(data.dtype)
    )


def test_output_path():
    assert str(output_path("a/labels.nii.gz")) == "a/labels_cleaned.nii.gz"
    assert str(output_path("a/labels.nii", "_x")) == "a/labels_x.nii"
    with pytest.raises(ValueError):
        output_path("a/labels.mgz")


def write_labels(path, data, scale=2.0):
    affine = np.diag([scale, scale, scale, 1.0])
    nib.save(nib.Nifti1Image(data, affine), str(path))


def test_file_roundtrip_keeps_dtype_header_and_affine(tmp_path):
    data = synthetic_volume(2).astype(np.int16)
    path = tmp_path / "labels.nii.gz"
    write_labels(path, data)
    out, n_changed = clean_labels_file(path)
    assert out == tmp_path / "labels_cleaned.nii.gz" and n_changed > 0
    result = nib.load(str(out))
    assert result.get_data_dtype() == np.int16
    np.testing.assert_array_equal(result.affine, np.diag([2.0, 2.0, 2.0, 1.0]))
    np.testing.assert_array_equal(np.asanyarray(result.dataobj), reference_loop(data, SPARSE).astype(np.int16))


def test_expand_inputs_skips_outputs_and_reports_no_match(tmp_path):
    for name in ("a.nii.gz", "a_cleaned.nii.gz", "b.nii.gz"):
        (tmp_path / name).write_text("")
    files = expand_inputs([str(tmp_path / "*.nii.gz")])
    assert [f.name for f in files] == ["a.nii.gz", "b.nii.gz"]
    with pytest.raises(FileNotFoundError):
        expand_inputs([str(tmp_path / "*.mgz")])


def test_cli_default_labels_and_suffix(tmp_path, capsys):
    data = synthetic_volume(4)
    data[3, 3, 3] = 72
    for subject in ("s1", "s2"):
        (tmp_path / subject).mkdir()
        write_labels(tmp_path / subject / "labels.nii.gz", data)
    assert main(["clean-labels", str(tmp_path / "*" / "labels.nii.gz")]) == 0
    assert "Total changed voxels" in capsys.readouterr().out
    out = np.asanyarray(nib.load(str(tmp_path / "s1" / "labels_cleaned.nii.gz")).dataobj)
    assert not np.isin(out, SPARSE).any() and out[3, 3, 3] == 72
    # running again does not clean the cleaned files
    assert main(["clean-labels", str(tmp_path / "*" / "labels*.nii.gz")]) == 0
    assert not (tmp_path / "s1" / "labels_cleaned_cleaned.nii.gz").exists()


def test_cli_labels_option_and_custom_suffix(tmp_path):
    data = synthetic_volume(4)
    data[3, 3, 3] = 72
    write_labels(tmp_path / "l.nii.gz", data)
    assert main(["clean-labels", str(tmp_path / "l.nii.gz"), "--labels", "72", "--suffix", "_no72"]) == 0
    out = np.asanyarray(nib.load(str(tmp_path / "l_no72.nii.gz")).dataobj)
    assert out[3, 3, 3] != 72 and np.isin(out, SPARSE).any()  # only 72 was removed


def test_cli_dry_run_writes_nothing(tmp_path, capsys):
    write_labels(tmp_path / "l.nii.gz", synthetic_volume(4))
    assert main(["clean-labels", str(tmp_path / "l.nii.gz"), "--dry-run"]) == 0
    assert "would change" in capsys.readouterr().out
    assert not (tmp_path / "l_cleaned.nii.gz").exists()


def test_cli_reports_errors(tmp_path, capsys):
    assert main(["clean-labels", str(tmp_path / "nothing*.nii.gz")]) == 1
    assert "error" in capsys.readouterr().err
