"""Tests de métricas, identificación y visualización."""

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dhts_py import dht2, dhtord, gauge, idht2, rdht
from dhts_py.utils import (
    coefficient_index,
    compare_coefficients,
    plot_coefficients,
    plot_reconstruction_comparison,
    reconstruction_metrics,
)


class TestUtils(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_coefficient_index(self):
        orders = dhtord(8, 3, "triangle")
        self.assertEqual(coefficient_index(orders, (2, 1)), 7)
        with self.assertRaises(ValueError):
            coefficient_index(orders, (4, 0))

    def test_reconstruction_metrics(self):
        original = np.array([[0.0, 1.0], [2.0, 3.0]])
        reconstructed = np.array([[0.0, 2.0], [2.0, 1.0]])
        metrics = reconstruction_metrics(original, reconstructed)
        self.assertEqual(metrics["mse"], 1.25)
        self.assertEqual(metrics["rmse"], np.sqrt(1.25))
        self.assertEqual(metrics["mae"], 0.75)
        self.assertEqual(metrics["max_abs_error"], 2.0)
        self.assertTrue(np.isfinite(metrics["psnr"]))
        self.assertEqual(reconstruction_metrics(original, original)["psnr"], float("inf"))

    def test_compare_coefficients_per_order(self):
        orders = [(0, 0), (1, 0)]
        first = np.zeros((2, 3, 2))
        second = first.copy()
        second[..., 1] = 2.0
        comparison = compare_coefficients(first, second, orders)
        self.assertEqual(comparison[(0, 0)]["mse"], 0.0)
        self.assertEqual(comparison[(1, 0)]["mse"], 4.0)
        self.assertEqual(comparison[(1, 0)]["max_abs_error"], 2.0)

    def test_plot_coefficients_preserves_input_and_layouts(self):
        rng = np.random.default_rng(9)
        for region in ("triangle", "square"):
            orders = dhtord(3, 2, region)
            coefficients = rng.normal(size=(4, 5, len(orders)))
            original = coefficients.copy()
            figure, axes = plot_coefficients(coefficients, orders, region, "individual")
            np.testing.assert_array_equal(coefficients, original)
            self.assertIsNotNone(figure)
            if region == "square":
                self.assertEqual(axes.shape, (3, 3))
                self.assertEqual(axes[0, 2].get_title(), "$L_{2,0}$")
                self.assertEqual(axes[2, 0].get_title(), "$L_{0,2}$")

    def test_plot_reconstruction_comparison(self):
        original = np.arange(20, dtype=np.float64).reshape(4, 5)
        figure, axes = plot_reconstruction_comparison(original, original + 0.1)
        self.assertIsNotNone(figure)
        self.assertEqual(len(axes), 3)

    def test_readme_workflows(self):
        image = np.random.default_rng(1).normal(size=(12, 10))
        triangle_orders = dhtord(8, 3, "triangle")
        triangle = dht2(image, 8, 3, 1, "triangle")
        plot_coefficients(triangle, triangle_orders, "triangle")

        square_orders = dhtord(8, 3, "square")
        square = dht2(image, 8, 3, 1, "square")
        plot_coefficients(square, square_orders, "square")

        complete = dht2(image, 8, 8, 1, "square")
        reconstructed = idht2(complete, image.shape, 8, 8, 1, "square")
        self.assertLess(reconstruction_metrics(image, reconstructed)["max_abs_error"], 3e-13)

        theta = gauge(triangle, 8, 3, "gradient", "triangle")
        rotated = rdht(triangle, theta, 8, 3, "forward", "triangle")
        recovered = rdht(rotated, theta, 8, 3, "inverse", "triangle")
        np.testing.assert_allclose(recovered, triangle, atol=5e-13, rtol=5e-13)


if __name__ == "__main__":
    unittest.main()
