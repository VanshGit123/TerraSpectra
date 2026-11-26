import unittest

import numpy as np

from app.preprocessing import (
    get_spectral_summary,
    normalize_cube,
    standardize_bands,
)

class TestNormalizeCube(unittest.TestCase):
    def test_scales_values_to_zero_and_one(self):
        cube = np.array([[[2.0, 4.0], [6.0, 8.0]]])
        result = normalize_cube(cube)

        self.assertEqual(result.dtype, np.float32)
        self.assertAlmostEqual(float(result.min()), 0.0)
        self.assertAlmostEqual(float(result.max()), 1.0)

    def test_constant_cube_returns_zeros(self):
        cube = np.full((2, 2, 3), 7.0)
        result = normalize_cube(cube)

        np.testing.assert_array_equal(result, np.zeros_like(cube, dtype=np.float32))

    def test_rejects_non_three_dimensional_input(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            normalize_cube(np.ones((4, 4)))

    def test_rejects_non_finite_values(self):
        cube = np.ones((2, 2, 2))
        cube[0, 0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            normalize_cube(cube)

class TestStandardizeBands(unittest.TestCase):
    def test_each_band_has_zero_mean(self):
        cube = np.array(
            [
                [[1.0, 10.0], [3.0, 20.0]],
                [[5.0, 30.0], [7.0, 40.0]],
            ]
        )
        result = standardize_bands(cube)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result.mean(axis=(0, 1)), [0.0, 0.0], atol=1e-6)

    def test_constant_band_remains_finite_and_zero(self):
        cube = np.zeros((2, 2, 2), dtype=np.float32)
        cube[:, :, 1] = 5.0

        result = standardize_bands(cube)

        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_rejects_non_finite_values(self):
        cube = np.ones((2, 2, 2))
        cube[0, 0, 1] = np.inf

        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            standardize_bands(cube)

class TestSpectralSummary(unittest.TestCase):
    def test_returns_expected_summary_fields(self):
        cube = np.array([[[1.0, 3.0], [5.0, 7.0]]])
        summary = get_spectral_summary(cube)

        self.assertEqual(summary["shape"], [1, 2, 2])
        self.assertEqual(summary["band_count"], 2)
        self.assertAlmostEqual(summary["mean"], 4.0)
        self.assertAlmostEqual(summary["min"], 1.0)
        self.assertAlmostEqual(summary["max"], 7.0)
        self.assertGreater(summary["std"], 0.0)

    def test_rejects_non_three_dimensional_input(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            get_spectral_summary(np.ones((3, 3)))

if __name__ == "__main__":
    unittest.main()
