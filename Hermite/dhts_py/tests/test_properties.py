"""Mathematical-property tests; these are not MATLAB equivalence evidence."""

import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from dhts_py import dht2, dhtmtx, dhtord, gauge, idht2, rdht


class TestMathematicalProperties(unittest.TestCase):
    def test_dhtmtx_requested_sizes_are_float64_and_zero_dc_above_order_zero(self):
        for N, D in ((3, 3), (4, 4), (8, 3), (8, 8)):
            H, G = dhtmtx(N, D, 1)
            self.assertEqual(H.dtype, np.float64)
            self.assertEqual(G.dtype, np.float64)
            self.assertEqual(H.shape, (N + 1, D + 1))
            np.testing.assert_allclose(H[:, 1:].sum(axis=0), 0.0, atol=2e-15, rtol=0)

    def test_dhtord_complete_sequence_N8_D3(self):
        expected = np.array(
            [
                (0, 0),
                (1, 0), (0, 1),
                (2, 0), (1, 1), (0, 2),
                (3, 0), (2, 1), (1, 2), (0, 3),
            ]
        )
        np.testing.assert_array_equal(dhtord(8, 3, 2), expected)

    def test_constant_image_has_only_dc_interior(self):
        coefficients = dht2(np.ones((40, 40), dtype=np.float64), 8, 3, 1, "symm")
        interior = coefficients[8:-8, 8:-8]
        np.testing.assert_allclose(interior[..., 0], 1.0, atol=2e-15, rtol=0)
        self.assertLessEqual(float(np.max(np.abs(interior[..., 1:]))), 3e-15)

    def test_horizontal_ramp_responds_in_L10_with_positive_sign(self):
        ramp = np.tile(np.arange(40, dtype=np.float64), (40, 1))
        interior = dht2(ramp, 8, 3, 1, "symm")[8:-8, 8:-8]
        np.testing.assert_allclose(interior[..., 1], np.sqrt(2), atol=2e-14, rtol=0)
        np.testing.assert_allclose(interior[..., 2], 0.0, atol=2e-14, rtol=0)

    def test_vertical_ramp_responds_in_L01_with_positive_sign(self):
        ramp = np.tile(np.arange(40, dtype=np.float64)[:, None], (1, 40))
        interior = dht2(ramp, 8, 3, 1, "symm")[8:-8, 8:-8]
        np.testing.assert_allclose(interior[..., 1], 0.0, atol=2e-14, rtol=0)
        np.testing.assert_allclose(interior[..., 2], np.sqrt(2), atol=2e-14, rtol=0)

    def test_gauge_and_rdht_roundtrip(self):
        yy, xx = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32), indexing="ij")
        image = np.sin(2.3 * xx) + 0.4 * np.cos(1.7 * yy) + 0.2 * xx * yy
        cartesian = dht2(image, 8, 3, 1, "symm")
        theta = gauge(cartesian, 8, 3, 1)
        recovered = rdht(rdht(cartesian, 8, 3, "fwd", theta), 8, 3, "inv", theta)
        np.testing.assert_allclose(recovered, cartesian, atol=2e-13, rtol=2e-13)

    def test_house_transform_python_regression(self):
        house_path = Path(__file__).resolve().parents[2] / "Fusion_Images_ds" / "house.tif"
        with Image.open(house_path) as image:
            house = np.asarray(image, dtype=np.float64) / 255.0
        if house.ndim == 3 and house.shape[2] >= 3:
            house = (
                0.298936021293775 * house[..., 0]
                + 0.587043074451121 * house[..., 1]
                + 0.114020904255103 * house[..., 2]
            )
        elif house.ndim == 3:
            house = house[..., 0]
        coefficients = dht2(house, 8, 3, 1, "symm")
        self.assertEqual(coefficients.shape, house.shape + (10,))
        self.assertTrue(np.isfinite(coefficients).all())

    def test_complete_and_truncated_reconstruction_are_distinguished(self):
        yy, xx = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32), indexing="ij")
        image = np.sin(2.3 * xx) + 0.4 * np.cos(1.7 * yy) + 0.2 * xx * yy
        complete = idht2(dht2(image, 8, 16, 1), image.shape, 8, 16, 1)
        truncated = idht2(dht2(image, 8, 3, 1), image.shape, 8, 3, 1)
        np.testing.assert_allclose(complete, image, atol=3e-13, rtol=3e-13)
        self.assertGreater(float(np.mean((truncated - image) ** 2)), 1e-6)


if __name__ == "__main__":
    unittest.main()
