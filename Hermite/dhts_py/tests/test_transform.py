"""Tests del análisis, orientación, steering y síntesis."""

import unittest
from math import comb

import numpy as np

from dhts_py import dht2, dhtmtx, dhtord, gauge, idht2, rdht
from dhts_py.utils import reconstruction_metrics


class TestTransform(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260814)
        self.image = self.rng.normal(size=(17, 13))

    def test_dhtmtx_binomial_window_and_parity(self):
        for N in range(1, 9):
            filters = dhtmtx(N)
            expected_window = np.array([comb(N, index) for index in range(N + 1)]) / 2**N
            np.testing.assert_allclose(filters[:, 0], expected_window, atol=2e-15, rtol=0)
            for order in range(N + 1):
                np.testing.assert_allclose(
                    filters[::-1, order],
                    (-1) ** order * filters[:, order],
                    atol=2e-15,
                    rtol=0,
                )

    def test_dht2_shapes_and_float64(self):
        triangle = dht2(self.image, N=8, D=3, T=1, coefficient_region="triangle")
        square = dht2(self.image, N=8, D=3, T=2, coefficient_region="square")
        self.assertEqual(triangle.shape, (25, 21, 10))
        self.assertEqual(square.shape, (13, 11, 16))
        self.assertEqual(triangle.dtype, np.float64)
        self.assertEqual(square.dtype, np.float64)

    def test_full_square_reconstruction_T1(self):
        coefficients = dht2(self.image, 8, 8, 1, "square")
        reconstructed = idht2(coefficients, self.image.shape, 8, 8, 1, "square")
        np.testing.assert_allclose(reconstructed, self.image, atol=3e-13, rtol=3e-13)

    def test_full_square_reconstruction_with_subsampling(self):
        coefficients = dht2(self.image, 4, 4, 3, "square")
        reconstructed = idht2(coefficients, self.image.shape, 4, 4, 3, "square")
        np.testing.assert_allclose(reconstructed, self.image, atol=3e-13, rtol=3e-13)

    def test_triangle_reconstruction_is_truncated(self):
        coefficients = dht2(self.image, 8, 3, 1, "triangle")
        reconstructed = idht2(coefficients, self.image.shape, 8, 3, 1, "triangle")
        metrics = reconstruction_metrics(self.image, reconstructed)
        self.assertGreater(metrics["mse"], 1e-6)

    def test_square_synthesis_uses_terms_above_D_total(self):
        orders = dhtord(4, 4, "square")
        coefficients = dht2(self.image, 4, 4, 1, "square")
        complete = idht2(coefficients, self.image.shape, 4, 4, 1, "square")
        without_upper_corner = coefficients.copy()
        without_upper_corner[..., [i for i, order in enumerate(orders) if sum(order) > 4]] = 0.0
        truncated = idht2(without_upper_corner, self.image.shape, 4, 4, 1, "square")
        np.testing.assert_allclose(complete, self.image, atol=3e-13, rtol=3e-13)
        self.assertGreater(float(np.max(np.abs(truncated - complete))), 1e-4)

    def test_gradient_gauge(self):
        orders = dhtord(8, 3, "triangle")
        coefficients = np.zeros((3, 4, len(orders)), dtype=np.float64)
        coefficients[..., orders.index((1, 0))] = 1.0
        coefficients[..., orders.index((0, 1))] = 1.0
        theta = gauge(coefficients, 8, 3, "gradient", "triangle")
        np.testing.assert_allclose(theta, np.pi / 4, atol=2e-15, rtol=0)

    def test_hessian_gauge(self):
        orders = dhtord(8, 3, "triangle")
        coefficients = np.zeros((3, 4, len(orders)), dtype=np.float64)
        coefficients[..., orders.index((1, 0))] = 1.0
        coefficients[..., orders.index((2, 0))] = 1.0
        theta = gauge(coefficients, 8, 3, "hessian", "triangle")
        np.testing.assert_allclose(theta, 0.0, atol=2e-15, rtol=0)

    def test_rdht_first_order_sign_convention(self):
        orders = dhtord(8, 1, "triangle")
        coefficients = np.zeros((1, 1, len(orders)), dtype=np.float64)
        coefficients[..., orders.index((1, 0))] = 2.0
        coefficients[..., orders.index((0, 1))] = 3.0
        rotated = rdht(coefficients, np.pi / 2, 8, 1, "forward", "triangle")
        self.assertAlmostEqual(rotated[..., orders.index((1, 0))].item(), 3.0)
        self.assertAlmostEqual(rotated[..., orders.index((0, 1))].item(), -2.0)

    def test_rdht_roundtrip_triangle_and_square(self):
        for region, D in (("triangle", 3), ("square", 3), ("square", 8)):
            with self.subTest(region=region, D=D):
                coefficients = dht2(self.image, 8, D, 1, region)
                theta = self.rng.uniform(-np.pi, np.pi, size=coefficients.shape[:2])
                rotated = rdht(coefficients, theta, 8, D, "forward", region)
                recovered = rdht(rotated, theta, 8, D, "inverse", region)
                np.testing.assert_allclose(recovered, coefficients, atol=5e-13, rtol=5e-13)

    def test_clear_validation_errors(self):
        with self.assertRaisesRegex(ValueError, "grayscale 2-D"):
            dht2(np.zeros((4, 4, 3)), 4, 2)
        with self.assertRaisesRegex(ValueError, "T debe"):
            dht2(np.zeros((4, 4)), 4, 2, 0)
        coefficients = dht2(np.zeros((4, 4)), 4, 2)
        with self.assertRaisesRegex(ValueError, "canales"):
            idht2(coefficients[..., :-1], (4, 4), 4, 2)
        with self.assertRaisesRegex(ValueError, "direction"):
            rdht(coefficients, 0.0, 4, 2, "sideways")


if __name__ == "__main__":
    unittest.main()
