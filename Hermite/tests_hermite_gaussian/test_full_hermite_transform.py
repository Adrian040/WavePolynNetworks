"""Numerical and structural tests for the Hermite--Gaussian transform."""

from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageOps

import full_hermite_transform_refactored as ht


def _expected_luminance(rgb):
    weights = np.asarray(
        [0.298936021293775, 0.587043074451121, 0.114020904255103]
    )
    return np.tensordot(
        np.asarray(rgb, dtype=np.float64)[..., :3], weights, axes=([-1], [0])
    )


class ImageReadingTests(unittest.TestCase):
    def test_grayscale_and_single_channel_preserve_intensities(self):
        grayscale = np.asarray([[0, 17, 255], [9, 81, 143]], dtype=np.uint8)
        direct = ht.read_image(grayscale)
        singleton = ht.read_image(grayscale[..., None])
        self.assertEqual(direct.dtype, np.float64)
        self.assertEqual(direct.ndim, 2)
        np.testing.assert_array_equal(direct, grayscale.astype(np.float64))
        np.testing.assert_array_equal(singleton, direct)

    def test_two_channels_and_la_tiff_ignore_alpha(self):
        luminance = np.asarray([[3, 27, 240], [91, 12, 188]], dtype=np.uint8)
        alpha = np.asarray([[0, 64, 255], [255, 1, 127]], dtype=np.uint8)
        luminance_alpha = np.dstack((luminance, alpha))
        np.testing.assert_array_equal(
            ht.read_image(luminance_alpha), luminance.astype(np.float64)
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "luminance_alpha.tif"
            Image.fromarray(luminance_alpha, mode="LA").save(path)
            loaded = ht.read_image(path)
        self.assertEqual(loaded.dtype, np.float64)
        self.assertEqual(loaded.ndim, 2)
        np.testing.assert_array_equal(loaded, luminance.astype(np.float64))

    def test_rgb_png_path_pil_and_array_use_identical_luminance(self):
        rgb = np.asarray(
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[12, 34, 56], [200, 100, 50], [1, 2, 3]],
            ],
            dtype=np.uint8,
        )
        expected = _expected_luminance(rgb)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.png"
            pil_image = Image.fromarray(rgb, mode="RGB")
            pil_image.save(path)
            from_path = ht.read_image(path)
            from_pil = ht.read_image(pil_image)
        from_array = ht.read_image(rgb)
        for result in (from_path, from_pil, from_array):
            self.assertEqual(result.dtype, np.float64)
            self.assertEqual(result.shape, rgb.shape[:2])
            np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12)

    def test_rgb_jpeg_path_matches_its_decoded_array(self):
        rng = np.random.default_rng(123)
        rgb = rng.integers(0, 256, size=(11, 13, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.jpg"
            Image.fromarray(rgb, mode="RGB").save(
                path, quality=100, subsampling=0
            )
            with Image.open(path) as decoded_image:
                decoded_rgb = np.asarray(decoded_image.convert("RGB"))
            from_path = ht.read_image(path)
        from_array = ht.read_image(decoded_rgb)
        np.testing.assert_allclose(from_path, from_array, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            from_path, _expected_luminance(decoded_rgb), rtol=0.0, atol=1e-12
        )

    def test_rgba_alpha_does_not_change_grayscale(self):
        rgb = np.asarray(
            [[[20, 50, 90], [250, 100, 5]], [[0, 1, 2], [80, 70, 60]]],
            dtype=np.uint8,
        )
        rgba_zero = np.dstack((rgb, np.zeros(rgb.shape[:2], dtype=np.uint8)))
        rgba_full = np.dstack((rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)))
        first = ht.read_image(rgba_zero)
        second = ht.read_image(rgba_full)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, _expected_luminance(rgb), atol=1e-12)

    def test_16_bit_grayscale_tiff_preserves_values(self):
        grayscale = np.asarray(
            [[0, 256, 1024], [4095, 32768, 65535]], dtype=np.uint16
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grayscale16.tif"
            Image.fromarray(grayscale).save(path)
            loaded = ht.read_image(path)
        self.assertEqual(loaded.dtype, np.float64)
        np.testing.assert_array_equal(loaded, grayscale.astype(np.float64))
        self.assertEqual(float(np.max(loaded)), 65535.0)

    def test_palette_and_cmyk_are_interpreted_as_rgb(self):
        palette = Image.new("P", (2, 1))
        palette.putpalette([255, 0, 0, 0, 255, 0] + [0] * (256 * 3 - 6))
        palette.putdata([0, 1])
        palette_rgb = np.asarray(palette.convert("RGB"))
        np.testing.assert_allclose(
            ht.read_image(palette), _expected_luminance(palette_rgb), atol=1e-12
        )

        cmyk = Image.new("CMYK", (2, 1), color=(10, 80, 140, 20))
        cmyk_rgb = np.asarray(cmyk.convert("RGB"))
        np.testing.assert_allclose(
            ht.read_image(cmyk), _expected_luminance(cmyk_rgb), atol=1e-12
        )

    def test_exif_orientation_is_applied_before_grayscale(self):
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]
        rgb[1, 2] = [0, 255, 0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oriented.jpg"
            exif = Image.Exif()
            exif[274] = 6
            Image.fromarray(rgb, mode="RGB").save(
                path, quality=100, subsampling=0, exif=exif
            )
            with Image.open(path) as encoded:
                oriented_rgb = np.asarray(
                    ImageOps.exif_transpose(encoded).convert("RGB")
                )
            loaded = ht.read_image(path)
        self.assertEqual(loaded.shape, (3, 2))
        np.testing.assert_allclose(
            loaded, _expected_luminance(oriented_rgb), rtol=0.0, atol=1e-12
        )

    def test_invalid_arrays_raise_clear_errors(self):
        invalid_inputs = [
            np.zeros((4,)),
            np.zeros((2, 3, 5)),
            np.zeros((2, 3), dtype=np.complex128),
            np.asarray([["not", "numeric"]]),
            np.asarray([[np.nan]]),
        ]
        for invalid in invalid_inputs:
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaises(ValueError):
                    ht.read_image(invalid)


class OrderAndFilterTests(unittest.TestCase):
    def test_channel_order(self):
        expected_triangle = [
            (0, 0),
            (1, 0),
            (0, 1),
            (2, 0),
            (1, 1),
            (0, 2),
            (3, 0),
            (2, 1),
            (1, 2),
            (0, 3),
        ]
        self.assertEqual(ht.hermite_orders(3, "triangle"), expected_triangle)

        square = ht.hermite_orders(3, "square")
        self.assertEqual(len(square), 16)
        self.assertEqual(square[:10], expected_triangle)
        self.assertEqual(
            square,
            sorted(square, key=lambda order: (sum(order), -order[0])),
        )

    def test_filter_parity_and_automatic_support(self):
        bank = ht.build_hermite_gaussian_filter_bank(3, 2.0, "square")
        self.assertGreater(bank["support_radius"], 4)
        self.assertEqual(bank["kernel_size"] % 2, 1)
        self.assertEqual(bank["kernel_size"], 2 * bank["support_radius"] + 1)

        radius = bank["support_radius"]
        tail_coordinates = np.arange(radius, radius + bank["tail_samples"])
        for order, values in bank["analysis_filters_1d"].items():
            np.testing.assert_allclose(
                values,
                ((-1) ** order) * values[::-1],
                rtol=0.0,
                atol=5e-15,
            )
            tail = np.abs(
                ht._analysis_filter_values(order, tail_coordinates, 2.0)
            )
            peak = np.max(np.abs(values))
            self.assertTrue(np.all(tail / peak < bank["tail_tolerance"]))

    def test_constant_image_has_negligible_nondc_response(self):
        image = np.full((45, 47), 7.0)
        bank = ht.build_hermite_gaussian_filter_bank(3, 2.0, "triangle")
        coefficients = ht.cartesian_hermite_transform(image, bank)
        dc_scale = float(np.max(np.abs(coefficients[(0, 0)])))
        self.assertGreater(dc_scale, 0.0)
        for order in [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]:
            leakage = float(np.max(np.abs(coefficients[order]))) / dc_scale
            self.assertLess(leakage, 1e-8)

    def test_m_and_n_follow_horizontal_and_vertical_axes(self):
        height, width = 49, 51
        horizontal = np.broadcast_to(np.arange(width), (height, width)).astype(float)
        vertical = np.broadcast_to(np.arange(height)[:, None], (height, width)).astype(float)
        bank = ht.build_hermite_gaussian_filter_bank(1, 2.0, "triangle")
        radius = bank["support_radius"]
        interior = np.s_[radius:-radius, radius:-radius]

        horizontal_coefficients = ht.cartesian_hermite_transform(horizontal, bank)
        vertical_coefficients = ht.cartesian_hermite_transform(vertical, bank)
        h_x = np.mean(np.abs(horizontal_coefficients[(1, 0)][interior]))
        h_y = np.max(np.abs(horizontal_coefficients[(0, 1)][interior]))
        v_x = np.max(np.abs(vertical_coefficients[(1, 0)][interior]))
        v_y = np.mean(np.abs(vertical_coefficients[(0, 1)][interior]))
        self.assertGreater(h_x, 1e6 * max(h_y, np.finfo(float).eps))
        self.assertGreater(v_y, 1e6 * max(v_x, np.finfo(float).eps))

    def test_transform_is_linear_and_preserves_input_scale(self):
        rng = np.random.default_rng(42)
        image = rng.normal(size=(25, 27))
        scale = 37.5
        bank = ht.build_hermite_gaussian_filter_bank(2, 2.0, "triangle")
        first = ht.cartesian_hermite_transform(image, bank)
        scaled = ht.cartesian_hermite_transform(scale * image, bank)
        for order in bank["orders"]:
            np.testing.assert_allclose(
                scaled[order], scale * first[order], rtol=2e-14, atol=2e-13
            )
        loaded = ht.read_image(np.full((3, 4), 200, dtype=np.uint8))
        self.assertEqual(float(loaded[0, 0]), 200.0)

    def test_small_sigma_emits_discretization_warning(self):
        with self.assertWarnsRegex(RuntimeWarning, "DC leakage"):
            ht.build_hermite_gaussian_filter_bank(3, 0.5, "triangle")


class SteeringTests(unittest.TestCase):
    def test_first_order_dominant_steering(self):
        shape = (4, 5)
        coefficients = {
            (0, 0): np.zeros(shape),
            (1, 0): np.full(shape, 3.0),
            (0, 1): np.full(shape, 4.0),
        }
        theta = ht.dominant_gradient_theta(coefficients)
        rotated = ht.rotate_hermite_coefficients(
            coefficients, theta, max_order=1, coefficient_region="triangle"
        )
        np.testing.assert_allclose(rotated[(1, 0)], 5.0, atol=1e-14)
        np.testing.assert_allclose(rotated[(0, 1)], 0.0, atol=1e-14)

    def test_triangle_steering_round_trip(self):
        rng = np.random.default_rng(7)
        orders = ht.hermite_orders(4, "triangle")
        coefficients = {order: rng.normal(size=(8, 9)) for order in orders}
        theta = rng.uniform(-np.pi, np.pi, size=(8, 9))
        rotated = ht.rotate_hermite_coefficients(
            coefficients, theta, 4, "triangle"
        )
        recovered = ht.inverse_rotate_hermite_coefficients(
            rotated, theta, 4, "triangle"
        )
        metrics = ht.coefficient_roundtrip_metrics(coefficients, recovered)
        self.assertLess(metrics["coefficient_max_abs_error"], 2e-14)

    def test_square_uses_complete_auxiliary_blocks(self):
        rng = np.random.default_rng(8)
        image = rng.normal(size=(21, 23))
        result = ht.hermite_transform_image(
            image,
            max_order=2,
            sigma=2.0,
            coefficient_region="square",
            use_rotation=True,
            use_inverse_rotation=True,
            use_inverse_transform=False,
            rotation_mode="fixed",
            angle=31.0,
        )
        expected_auxiliary = ht.hermite_orders(4, "triangle")
        self.assertEqual(result["auxiliary_orders"], expected_auxiliary)
        self.assertEqual(len(result["orders"]), 9)
        self.assertIn((4, 0), result["filter_bank"]["analysis_filters"])
        self.assertIn((0, 4), result["filter_bank"]["analysis_filters"])

        complete_cartesian = ht.cartesian_hermite_transform(
            image,
            result["filter_bank"],
            orders=expected_auxiliary,
        )
        complete_rotated = ht.rotate_hermite_coefficients(
            complete_cartesian,
            np.deg2rad(31.0),
            2,
            "square",
        )
        for order in result["orders"]:
            np.testing.assert_allclose(
                result["rotated_coefficients"][order],
                complete_rotated[order],
            )
        self.assertLess(
            result["coefficient_roundtrip_metrics"][
                "coefficient_max_abs_error"
            ],
            2e-14,
        )

        with self.assertRaisesRegex(ValueError, "auxiliary complete blocks"):
            ht.rotate_hermite_coefficients(
                result["cartesian_coefficients"],
                np.deg2rad(31.0),
                2,
                "square",
            )


class SynthesisAndWorkflowTests(unittest.TestCase):
    def test_visual_outputs_are_grayscale_except_theta(self):
        self.assertEqual(
            inspect.signature(ht.save_coefficients_grid).parameters[
                "cmap"
            ].default,
            "gray",
        )
        image = np.arange(9 * 11, dtype=float).reshape(9, 11)
        coefficients = {
            (0, 0): image.copy(),
            (1, 0): image.copy() - np.mean(image),
        }
        preserved = {order: values.copy() for order, values in coefficients.items()}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coefficient_path = root / "coefficients.png"
            comparison_path = root / "comparison.png"
            theta_path = root / "theta.png"
            original_path = root / "original.png"
            energy_path = root / "energy.png"
            ht.save_coefficients_grid(coefficients, coefficient_path, "Coefficients")
            ht.save_reconstruction_comparison(
                image, image + 1.0, comparison_path
            )
            ht.save_theta_image(
                np.linspace(-np.pi, np.pi, image.size).reshape(image.shape),
                theta_path,
            )
            result = ht.hermite_transform_image(
                image,
                max_order=1,
                coefficient_region="triangle",
                use_rotation=False,
                use_inverse_rotation=False,
                use_inverse_transform=False,
                output_paths={
                    "original_image": original_path,
                    "coefficient_energy": energy_path,
                },
            )

            for path in (
                coefficient_path,
                comparison_path,
                original_path,
                energy_path,
            ):
                rgb = np.asarray(Image.open(path).convert("RGB"), dtype=int)
                self.assertTrue(np.array_equal(rgb[..., 0], rgb[..., 1]), path)
                self.assertTrue(np.array_equal(rgb[..., 1], rgb[..., 2]), path)
            theta_rgb = np.asarray(Image.open(theta_path).convert("RGB"), dtype=int)
            self.assertTrue(
                np.any(theta_rgb[..., 0] != theta_rgb[..., 1])
                or np.any(theta_rgb[..., 1] != theta_rgb[..., 2])
            )

        for order in coefficients:
            np.testing.assert_array_equal(coefficients[order], preserved[order])
        np.testing.assert_array_equal(result["original_image"], image)

    def test_synthesis_shape_finiteness_and_sampling(self):
        rng = np.random.default_rng(11)
        image = rng.normal(size=(28, 31))
        result = ht.hermite_transform_image(
            image,
            max_order=1,
            sigma=2.0,
            coefficient_region="square",
            sampling_step=3,
            use_rotation=False,
            use_inverse_rotation=False,
            use_inverse_transform=True,
        )
        self.assertEqual(result["reconstructed_image"].shape, image.shape)
        self.assertTrue(np.isfinite(result["reconstructed_image"]).all())

        bank = ht.build_hermite_gaussian_filter_bank(1, 1.0, "triangle")
        coefficients = ht.cartesian_hermite_transform(
            image, bank, sampling_step=100
        )
        with self.assertRaisesRegex(ValueError, "without coverage"):
            ht.synthesize_hermite_image(
                coefficients, bank, image.shape, sampling_step=100
            )

    def test_more_orders_improve_controlled_smooth_case(self):
        y, x = np.mgrid[-1:1:41j, -1:1:43j]
        image = np.exp(-4.0 * (x**2 + y**2))
        low = ht.hermite_transform_image(
            image,
            max_order=0,
            sigma=2.0,
            coefficient_region="triangle",
            use_rotation=False,
            use_inverse_rotation=False,
            use_inverse_transform=True,
        )
        higher = ht.hermite_transform_image(
            image,
            max_order=2,
            sigma=2.0,
            coefficient_region="triangle",
            use_rotation=False,
            use_inverse_rotation=False,
            use_inverse_transform=True,
        )
        self.assertLess(
            higher["reconstruction_metrics"]["mse"],
            low["reconstruction_metrics"]["mse"],
        )

    def test_default_executes_complete_workflow(self):
        y, x = np.mgrid[-1:1:19j, -1:1:21j]
        result = ht.hermite_transform_image(np.exp(-(x**2 + y**2)))
        required = [
            "cartesian_coefficients",
            "rotated_coefficients",
            "recovered_cartesian_coefficients",
            "theta",
            "reconstructed_image",
            "coefficient_roundtrip_metrics",
            "reconstruction_metrics",
        ]
        for key in required:
            self.assertIsNotNone(result[key], key)
        self.assertEqual(result["coeff_stack"].shape[-1], len(result["orders"]))
        self.assertIs(result["transformed_image"], result["coefficient_energy"])

    def test_supported_flag_combinations(self):
        image = np.arange(15 * 17, dtype=float).reshape(15, 17)
        combinations = [
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (False, False, True),
            (True, False, True),
            (True, True, True),
        ]
        for use_rotation, use_inverse_rotation, use_inverse_transform in combinations:
            with self.subTest(
                rotation=use_rotation,
                inverse_rotation=use_inverse_rotation,
                inverse_transform=use_inverse_transform,
            ):
                result = ht.hermite_transform_image(
                    image,
                    max_order=2,
                    sigma=2.0,
                    coefficient_region="triangle",
                    use_rotation=use_rotation,
                    use_inverse_rotation=use_inverse_rotation,
                    use_inverse_transform=use_inverse_transform,
                )
                self.assertIsNotNone(result["cartesian_coefficients"])
                self.assertEqual(
                    result["rotated_coefficients"] is not None, use_rotation
                )
                self.assertEqual(
                    result["reconstructed_image"] is not None,
                    use_inverse_transform,
                )
                expected_recovered = use_rotation and (
                    use_inverse_rotation or use_inverse_transform
                )
                self.assertEqual(
                    result["recovered_cartesian_coefficients"] is not None,
                    expected_recovered,
                )

        with self.assertRaisesRegex(ValueError, "requires use_rotation"):
            ht.hermite_transform_image(
                image,
                use_rotation=False,
                use_inverse_rotation=True,
                use_inverse_transform=False,
            )

    def test_public_api_and_source_have_no_algebraic_inverse_controls(self):
        signature = inspect.signature(ht.hermite_transform_image)
        removed_parameters = {
            "kernel" + "_size",
            "exact" + "_reconstruction",
            "r" + "cond",
        }
        self.assertTrue(removed_parameters.isdisjoint(signature.parameters))
        source = inspect.getsource(ht)
        forbidden_fragments = [
            "np.linalg." + "pinv",
            "analysis" + "_matrix",
            "synthesis" + "_matrix",
            "exact" + "_reconstruction",
            "r" + "cond",
        ]
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
