"""Numerical and structural tests for the Hermite--Gaussian transform."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

import full_hermite_transform_refactored as ht


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
