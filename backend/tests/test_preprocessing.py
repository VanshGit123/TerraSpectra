
import unittest

import numpy as np

from app.data import make_synthetic_cube
from app.preprocessing import normalize_cube, standardize_bands
from app.pca import reduce_dimensions


class TestHyperspectralPreprocessing(unittest.TestCase):

    def setUp(self):
        self.cube, self.mask = make_synthetic_cube(
            height=16,
            width=16,
            bands=8,
            seed=42,
        )

    def test_synthetic_cube_shape(self):
        self.assertEqual(self.cube.shape, (16, 16, 8))
        self.assertEqual(self.mask.shape, (16, 16))

    def test_synthetic_data_is_reproducible(self):
        cube2, mask2 = make_synthetic_cube(
            height=16,
            width=16,
            bands=8,
            seed=42,
        )

        np.testing.assert_array_equal(self.cube, cube2)
        np.testing.assert_array_equal(self.mask, mask2)

    def test_normalization_range(self):
        normalized = normalize_cube(self.cube)

        self.assertEqual(normalized.shape, self.cube.shape)
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))

    def test_standardization_per_band(self):
        standardized = standardize_bands(self.cube)

        self.assertEqual(standardized.shape, self.cube.shape)

        band_means = np.mean(standardized, axis=(0, 1))
        np.testing.assert_allclose(
            band_means,
            np.zeros(8),
            atol=1e-5,
        )

    def test_pca_reduces_spectral_dimensions(self):
        reduced, variance = reduce_dimensions(
            self.cube,
            n_components=4,
        )

        self.assertEqual(reduced.shape, (16, 16, 4))
        self.assertEqual(variance.shape, (4,))
        self.assertTrue(np.all(variance >= 0))
        self.assertLessEqual(float(np.sum(variance)), 1.0 + 1e-6)

    def test_invalid_normalization_input(self):
        with self.assertRaises(ValueError):
            normalize_cube(np.zeros((16, 16)))

    def test_invalid_pca_components(self):
        with self.assertRaises(ValueError):
            reduce_dimensions(self.cube, n_components=0)


if __name__ == "__main__":
    unittest.main()
