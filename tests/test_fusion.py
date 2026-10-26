import numpy as np
import pytest

from src.multi_atlas.multi_atlas_fusion import (
    get_multi_atlas_proba_seg,
    get_multi_atlas_tissue_prior,
    get_structure_seg_from_tissue_seg,
)


def reference_proba_seg(warped_atlases, weights, labels):
    """Straightforward numpy version of get_multi_atlas_proba_seg."""
    shape = warped_atlases.shape[1:]
    out = np.zeros((*shape, len(labels)), dtype=np.float64)
    any_label = warped_atlases.sum(axis=0) > 0
    for idx, label in enumerate(labels):
        out[..., idx] = (weights * (warped_atlases == label)).sum(axis=0)
    out[~any_label] = 0
    out[~any_label, 0] = 1
    empty = out.sum(axis=-1) == 0
    out[empty, 0] = 1
    return out


def make_atlases(rng, n_atlases=4, shape=(6, 7, 5), labels=(0, 3, 8, 21)):
    warped = rng.choice(labels, size=(n_atlases, *shape)).astype(np.uint16)
    weights = rng.random((n_atlases, *shape)).astype(np.float32)
    return warped, weights, np.array(labels)


def test_proba_seg_matches_reference(rng):
    warped, weights, labels = make_atlases(rng)
    out = get_multi_atlas_proba_seg(warped, weights, labels, np.zeros((*warped.shape[1:], len(labels)), np.float32))
    np.testing.assert_allclose(out, reference_proba_seg(warped, weights, labels), rtol=1e-5, atol=1e-6)


def test_proba_seg_background_where_no_atlas_has_a_label(rng):
    warped, weights, labels = make_atlases(rng)
    warped[:, 0, 0, 0] = 0
    out = get_multi_atlas_proba_seg(warped, weights, labels, np.zeros((*warped.shape[1:], len(labels)), np.float32))
    assert out[0, 0, 0, 0] == 1 and out[0, 0, 0, 1:].sum() == 0


def test_proba_seg_is_sum_of_weights_of_agreeing_atlases():
    warped = np.array([[[[5]]], [[[5]]], [[[7]]]], dtype=np.uint16)  # 3 atlases, 1 voxel
    weights = np.array([[[[0.25]]], [[[0.5]]], [[[0.125]]]], dtype=np.float32)
    labels = np.array([0, 5, 7])
    out = get_multi_atlas_proba_seg(warped, weights, labels, np.zeros((1, 1, 1, 3), np.float32))
    np.testing.assert_allclose(out[0, 0, 0], [0.0, 0.75, 0.125])


STRUCTURE_DICT = {"tissues": {0: [0], 3: [1], 8: [1, 2], 21: [2]}}


def test_tissue_prior_is_normalised_and_splits_over_assigned_tissues():
    proba = np.zeros((2, 1, 1, 4), dtype=np.float32)
    proba[0, 0, 0] = [0, 0, 1, 0]  # label 8 -> tissues 1 and 2
    proba[1, 0, 0] = [0, 0.5, 0, 0.5]  # label 3 -> tissue 1, label 21 -> tissue 2
    prior = get_multi_atlas_tissue_prior(proba, np.array([0, 3, 8, 21]), STRUCTURE_DICT, num_tissue=3)
    np.testing.assert_allclose(prior.sum(axis=-1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(prior[0, 0, 0], [0, 0.5, 0.5])
    np.testing.assert_allclose(prior[1, 0, 0], [0, 0.5, 0.5])


def test_tissue_prior_background_for_unassigned_and_masked_voxels():
    proba = np.zeros((2, 1, 1, 4), dtype=np.float32)
    proba[0, 0, 0] = [0, 1, 0, 0]
    proba[1, 0, 0] = [0, 1, 0, 0]
    mask = np.array([[[1]], [[0]]])
    prior = get_multi_atlas_tissue_prior(proba, np.array([0, 3, 8, 21]), STRUCTURE_DICT, num_tissue=3, mask=mask)
    np.testing.assert_allclose(prior[0, 0, 0], [0, 1, 0])
    np.testing.assert_allclose(prior[1, 0, 0], [0.5, 0.5, 0])  # background forced to 1, then renormalised
    # a label that is not in the structure dictionary contributes nothing -> background
    prior = get_multi_atlas_tissue_prior(proba, np.array([0, 99, 8, 21]), STRUCTURE_DICT, num_tissue=3)
    np.testing.assert_allclose(prior[..., 0], 1.0)


def test_tissue_prior_removes_values_below_one_percent():
    proba = np.zeros((1, 1, 1, 4), dtype=np.float32)
    proba[0, 0, 0] = [0, 0.995, 0, 0.005]
    prior = get_multi_atlas_tissue_prior(proba, np.array([0, 3, 8, 21]), STRUCTURE_DICT, num_tissue=3)
    np.testing.assert_allclose(prior[0, 0, 0], [0, 1, 0])


def nested_loop_reference(tiss_seg, lab_probs, labels, tissue_dict):
    out = np.zeros(tiss_seg.shape, dtype=np.int32)
    order = np.argsort(-lab_probs, axis=-1)
    for idx in np.ndindex(tiss_seg.shape):
        for lab_idx in order[idx]:
            if tiss_seg[idx] in tissue_dict[labels[lab_idx]]:
                out[idx] = labels[lab_idx]
                break
    return out


def test_structure_seg_matches_nested_loop_reference(rng):
    shape, n_labels, n_tissues = (5, 6, 4), 12, 5
    tiss_seg = rng.integers(0, n_tissues, size=shape).astype(np.uint16)
    lab_probs = rng.random((*shape, n_labels))
    lab_probs /= lab_probs.sum(axis=-1, keepdims=True)
    labels = np.arange(100, 100 + n_labels)
    tissue_dict = {int(label): list(rng.choice(n_tissues, size=rng.integers(1, 4), replace=False)) for label in labels}
    result = get_structure_seg_from_tissue_seg(tiss_seg, lab_probs, labels, tissue_dict)
    np.testing.assert_array_equal(result, nested_loop_reference(tiss_seg, lab_probs, labels, tissue_dict))


def test_structure_seg_picks_most_probable_label_compatible_with_tissue():
    tiss_seg = np.array([[[1]]], dtype=np.uint16)
    lab_probs = np.array([[[[0.5, 0.3, 0.2]]]])
    labels = np.array([10, 20, 30])
    result = get_structure_seg_from_tissue_seg(tiss_seg, lab_probs, labels, {10: [2], 20: [1], 30: [1]})
    assert result[0, 0, 0] == 20


def test_structure_seg_label_without_tissue_entry_stops_the_search():
    tiss_seg = np.array([[[1]]], dtype=np.uint16)
    lab_probs = np.array([[[[0.6, 0.4]]]])
    result = get_structure_seg_from_tissue_seg(tiss_seg, lab_probs, np.array([10, 20]), {20: [1]})
    assert result[0, 0, 0] == 0


def test_structure_seg_accepts_unsigned_tissue_segmentations():
    # np.ones_like(uint16) * -1 raises an OverflowError with NumPy 2
    result = get_structure_seg_from_tissue_seg(
        np.zeros((2, 2, 2), dtype=np.uint16), np.ones((2, 2, 2, 1)), np.array([5]), {5: [3]}
    )
    assert result.dtype == np.int32 and (result == 0).all()


@pytest.mark.parametrize("n_atlases", [1, 3])
def test_proba_seg_runs_for_any_number_of_atlases(rng, n_atlases):
    warped, weights, labels = make_atlases(rng, n_atlases=n_atlases)
    out = get_multi_atlas_proba_seg(warped, weights, labels, np.zeros((*warped.shape[1:], len(labels)), np.float32))
    assert out.shape == (*warped.shape[1:], len(labels))
