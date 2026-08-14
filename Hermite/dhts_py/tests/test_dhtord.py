"""Tests de selección y orden de coefficient maps."""

import unittest

from dhts_py import dhtord


class TestDhtord(unittest.TestCase):
    def test_triangle_N8_D3(self):
        expected = [
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
        self.assertEqual(dhtord(8, 3, "triangle"), expected)

    def test_square_N8_D3_has_16_orders(self):
        orders = dhtord(8, 3, "square")
        self.assertEqual(len(orders), 16)
        self.assertEqual(set(orders), {(m, n) for m in range(4) for n in range(4)})
        self.assertEqual(orders[:6], [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)])

    def test_full_square_N8_D8_has_81_orders(self):
        orders = dhtord(8, 8, "square")
        self.assertEqual(len(orders), 81)
        self.assertIn((8, 0), orders)
        self.assertIn((0, 8), orders)
        self.assertIn((8, 8), orders)

    def test_square_clips_D_to_N(self):
        self.assertEqual(dhtord(3, 9, "square"), dhtord(3, 3, "square"))

    def test_invalid_parameters(self):
        for arguments in [(-1, 0, "triangle"), (4, -1, "triangle"), (4, 9, "triangle")]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                dhtord(*arguments)
        with self.assertRaisesRegex(ValueError, "coefficient_region"):
            dhtord(4, 2, "circle")
        with self.assertRaises(ValueError):
            dhtord(True, 0, "triangle")


if __name__ == "__main__":
    unittest.main()
