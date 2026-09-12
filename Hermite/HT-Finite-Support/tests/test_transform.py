"""Pruebas del núcleo y del flujo completo Hermite--Gaussiano."""

from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from hermite_transform import (  # noqa: E402
    build_filter_bank,
    cartesian_transform,
    coefficient_roundtrip_metrics,
    dominant_theta,
    hermite_orders,
    hermite_transform_image,
    inverse_rotate_coefficients,
    read_image,
    rotate_coefficients,
    synthesize,
)


class ImageAndFilterTests(unittest.TestCase):
    """Comprueba lectura, órdenes y muestreo de filtros."""

    def test_16_bit_tiff_preserves_values(self):
        image = np.asarray(
            [[0, 256, 1024], [4095, 32768, 65535]],
            dtype=np.uint16,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image16.tif"
            Image.fromarray(image).save(path)
            loaded = read_image(path)
        np.testing.assert_array_equal(loaded, image.astype(np.float64))

    def test_orders_and_complete_square_steering_region(self):
        self.assertEqual(
            hermite_orders(2, "triangle"),
            [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)],
        )
        bank = build_filter_bank(2, 2.0, "square")
        self.assertEqual(len(bank["orders"]), 9)
        self.assertEqual(bank["steering_orders"], hermite_orders(4, "triangle"))

    def test_filter_parity_and_small_sigma_warning(self):
        bank = build_filter_bank(3, 2.0, "triangle")
        for order, values in bank["filters"].items():
            np.testing.assert_allclose(
                values,
                ((-1) ** order) * values[::-1],
                rtol=0.0,
                atol=5e-15,
            )
            self.assertLess(
                bank["diagnostics"][order]["edge_relative_magnitude"],
                bank["tail_tolerance"],
            )
        with self.assertWarnsRegex(RuntimeWarning, "fuga DC"):
            build_filter_bank(3, 0.5, "triangle")


class AnalysisAndSteeringTests(unittest.TestCase):
    """Comprueba ejes, bordes y propiedades del steering."""

    def test_axis_convention_and_boundary_modes(self):
        height, width = 31, 33
        horizontal = np.broadcast_to(np.arange(width), (height, width)).astype(float)
        bank = build_filter_bank(1, 2.0, "triangle")
        coefficients = cartesian_transform(horizontal, bank)
        radius = bank["radius"]
        interior = np.s_[radius:-radius, radius:-radius]
        x_response = np.mean(np.abs(coefficients[(1, 0)][interior]))
        y_response = np.max(np.abs(coefficients[(0, 1)][interior]))
        self.assertGreater(x_response, 1e6 * max(y_response, np.finfo(float).eps))

        for boundary in ("symmetric", "edge", "constant", "wrap"):
            maps = cartesian_transform(horizontal, bank, boundary=boundary)
            self.assertEqual(maps[(0, 0)].shape, horizontal.shape)

    def test_first_order_alignment_and_roundtrip(self):
        shape = (6, 7)
        coefficients = {
            (0, 0): np.zeros(shape),
            (1, 0): np.full(shape, 3.0),
            (0, 1): np.full(shape, 4.0),
        }
        orders = hermite_orders(1, "triangle")
        theta = dominant_theta(coefficients)
        rotated = rotate_coefficients(coefficients, theta, orders)
        recovered = inverse_rotate_coefficients(rotated, theta, orders)
        np.testing.assert_allclose(rotated[(1, 0)], 5.0, atol=1e-14)
        np.testing.assert_allclose(rotated[(0, 1)], 0.0, atol=1e-14)
        metrics = coefficient_roundtrip_metrics(coefficients, recovered)
        self.assertLess(metrics["coefficient_max_abs_error"], 2e-14)

    def test_random_higher_order_roundtrip(self):
        rng = np.random.default_rng(19)
        orders = hermite_orders(4, "triangle")
        coefficients = {order: rng.normal(size=(8, 9)) for order in orders}
        theta = rng.uniform(-np.pi, np.pi, size=(8, 9))
        rotated = rotate_coefficients(coefficients, theta, orders)
        recovered = inverse_rotate_coefficients(rotated, theta, orders)
        metrics = coefficient_roundtrip_metrics(coefficients, recovered)
        self.assertLess(metrics["coefficient_max_abs_error"], 2e-14)


class SynthesisAndWorkflowTests(unittest.TestCase):
    """Comprueba síntesis, resultados y nombres de archivo."""

    def test_synthesis_is_finite_and_higher_order_improves_smooth_image(self):
        y, x = np.mgrid[-1:1:41j, -1:1:43j]
        image = np.exp(-4.0 * (x**2 + y**2))
        errors = []
        for order in (0, 2):
            bank = build_filter_bank(order, 2.0, "triangle")
            coefficients = cartesian_transform(image, bank)
            reconstructed = synthesize(coefficients, bank, image.shape)
            self.assertTrue(np.isfinite(reconstructed).all())
            errors.append(float(np.mean((image - reconstructed) ** 2)))
        self.assertLess(errors[1], errors[0])

    def test_default_workflow_writes_nine_descriptive_results(self):
        rng = np.random.default_rng(23)
        image = rng.normal(size=(19, 21))
        expected = {
            "01_original_input_grayscale_image.png",
            "02_cartesian_hermite_coefficient_maps.png",
            "03_steered_rotated_hermite_coefficient_maps.png",
            "04_local_gradient_orientation_theta_in_degrees.png",
            "05_recovered_cartesian_from_rotated_coefficients.png",
            "06_truncated_hermite_synthesis_reconstructed_image.png",
            "07_reconstruction_comparison_with_absolute_error.png",
            "08_rotated_hermite_coefficient_energy_without_dc.png",
            "09_steering_roundtrip_and_reconstruction_metrics.csv",
        }
        with tempfile.TemporaryDirectory() as directory:
            result = hermite_transform_image(
                image,
                max_order=2,
                sampling_step=2,
                results_path=Path(directory) / "nested" / "results",
            )
            generated = {path.name for path in result["saved_files"].values()}
            self.assertEqual(generated, expected)
            self.assertTrue(
                all(path.is_file() for path in result["saved_files"].values())
            )
        self.assertEqual(result["reconstructed_image"].shape, image.shape)
        self.assertLess(
            result["coefficient_roundtrip_metrics"][
                "coefficient_max_abs_error"
            ],
            2e-14,
        )


if __name__ == "__main__":
    unittest.main()
