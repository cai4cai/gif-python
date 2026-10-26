import numpy as np
import pytest

from src.multi_atlas.utils import gaussian_smooth_with_mask, get_lncc_distance, nibabel_load_and_get_fdata
from src.utils.config import GifConfig


def smooth_image(rng, shape=(24, 24, 24)):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(rng.normal(size=shape), 1.5).astype(np.float32) * 100


def test_masked_smoothing_ignores_values_outside_the_mask():
    image = np.ones((16, 16, 16), dtype=np.float32)
    mask = np.zeros_like(image)
    mask[4:12, 4:12, 4:12] = 1
    image[mask == 0] = 1000  # must not leak into the masked region
    smoothed = gaussian_smooth_with_mask(image, mask, sigma=1.5)
    np.testing.assert_allclose(smoothed[mask == 1], 1.0, rtol=1e-4)
    assert (smoothed[mask == 0] == 0).all()


def test_lncc_distance_is_zero_for_identical_images(rng):
    image = smooth_image(rng)
    distance = get_lncc_distance(image, None, image.copy(), spacing=(1, 1, 1))
    assert distance.shape == image.shape
    assert np.nanmax(np.abs(distance)) < 1e-3


def test_lncc_distance_is_two_for_inverted_image(rng):
    image = smooth_image(rng)
    distance = get_lncc_distance(image, None, -image, spacing=(1, 1, 1))
    np.testing.assert_allclose(np.nanmedian(distance), 2.0, atol=1e-3)


def test_lncc_distance_orders_similar_before_dissimilar(rng):
    image = smooth_image(rng)
    similar = image + 0.1 * smooth_image(rng)
    different = smooth_image(rng)
    d_similar = np.nanmean(get_lncc_distance(image, None, similar, spacing=(1, 1, 1)))
    d_different = np.nanmean(get_lncc_distance(image, None, different, spacing=(1, 1, 1)))
    assert d_similar < d_different


def test_lncc_distance_nan_where_variance_vanishes():
    image = np.zeros((10, 10, 10), dtype=np.float32)
    distance = get_lncc_distance(image, None, image, spacing=(1, 1, 1))
    assert np.isnan(distance).all()


def test_lncc_kernel_negative_values_are_millimetres_positive_values_are_voxels(rng):
    image = smooth_image(rng)
    other = smooth_image(rng)
    # -2 (mm) at a spacing of 2 mm is 1 voxel; +1 is 1 voxel
    negative = get_lncc_distance(image, None, other, (2, 2, 2), GifConfig(lncc_sigma=(-2.0, -2.0, -2.0)))
    positive = get_lncc_distance(image, None, other, (2, 2, 2), GifConfig(lncc_sigma=(1.0, 1.0, 1.0)))
    np.testing.assert_allclose(negative, positive, equal_nan=True, rtol=1e-5)
    wider = get_lncc_distance(image, None, other, (2, 2, 2), GifConfig(lncc_sigma=(3.0, 3.0, 3.0)))
    assert not np.allclose(positive, wider, equal_nan=True)


def test_lncc_respects_mask(rng):
    image = smooth_image(rng)
    other = smooth_image(rng)
    mask = np.zeros(image.shape, dtype=np.uint16)
    mask[4:20, 4:20, 4:20] = 1
    masked = get_lncc_distance(image, mask, other, (1, 1, 1))
    assert np.isnan(masked[0, 0, 0]) or masked[0, 0, 0] == 1.0  # outside the mask: 0 / 0 or lncc 0
    assert np.isfinite(masked[mask == 1]).any()


def test_weights_from_distance_are_monotonic(rng):
    temperature = GifConfig().weights_temperature
    distances = np.array([0.0, 0.5, 1.0, 2.0])
    weights = np.exp(-distances / temperature)
    assert weights[0] == 1.0 and (np.diff(weights) < 0).all()


@pytest.mark.parametrize("dtype", [np.float32, np.uint16])
def test_nibabel_load_and_get_fdata(tmp_path, write_nifti, dtype):
    data = (np.arange(24).reshape(2, 3, 4) * 3).astype(np.uint16)
    path = write_nifti(data, tmp_path / "x.nii.gz")
    loaded = nibabel_load_and_get_fdata(path, dtype)
    assert loaded.dtype == dtype
    np.testing.assert_array_equal(loaded, data)
