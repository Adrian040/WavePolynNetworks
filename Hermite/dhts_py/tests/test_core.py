import unittest
from math import comb
from pathlib import Path

import numpy as np

import dhts_py as dhts


class TestCore(unittest.TestCase):
    def test_hermite_polynomials(self):
        x = np.array([-1.0, 0.0, 2.0])
        actual = dhts.hermite([0, 1, 2, 3], x)
        expected = np.column_stack((np.ones_like(x), 2*x, 4*x*x-2, 8*x**3-12*x))
        np.testing.assert_allclose(actual, expected, atol=0, rtol=0)

    def test_dhtmtx_binomial_and_parity(self):
        for N in range(1, 10):
            H = dhts.dhtmtx(N)
            expected_lowpass = np.array([comb(N, k) for k in range(N + 1)]) / 2**N
            np.testing.assert_allclose(H[:, 0], expected_lowpass, atol=1e-15)
            for order in range(N + 1):
                np.testing.assert_allclose(H[::-1, order], (-1)**order * H[:, order], atol=1e-15)

    def test_dhtord_matches_matlab_coordinate_order(self):
        expected = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])
        np.testing.assert_array_equal(dhts.dhtord(4, 2, 2), expected)

    def test_dht_idht_1d_full(self):
        rng = np.random.default_rng(11)
        for N, T in ((2, 1), (4, 2), (4, 3)):
            x = rng.normal(size=(29, 3))
            y = dhts.dht(x, N, N, T, 1)
            restored = dhts.idht(y, x.shape, N, N, T, 1)
            np.testing.assert_allclose(restored, x, atol=2e-13)

    def test_dht2_idht2_full(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(17, 13))
        for N, T in ((2, 1), (2, 2), (4, 3)):
            y = dhts.dht2(x, N, 2*N, T)
            restored = dhts.idht2(y, x.shape, N, 2*N, T)
            np.testing.assert_allclose(restored, x, atol=2e-13)

    def test_quincunx_dht2_idht2_full(self):
        x = np.random.default_rng(121).normal(size=(17, 13))
        y = dhts.dht2(x, 4, 8, 2, "q")
        restored = dhts.idht2(y, x.shape, 4, 8, 2, "q")
        np.testing.assert_allclose(restored, x, atol=2e-13)

    def test_dht3_idht3_full(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(5, 6, 7))
        y = dhts.dht3(x, 2, 6, 2)
        restored = dhts.idht3(y, x.shape, 2, 6, 2)
        np.testing.assert_allclose(restored, x, atol=3e-13)

    def test_rdht_roundtrip_with_explicit_angle(self):
        rng = np.random.default_rng(14)
        N, D = 5, 7
        y = rng.normal(size=(4, 3, len(dhts.dhtord(N, D, 2))))
        theta = rng.uniform(-np.pi, np.pi, size=y.shape[:2])
        restored = dhts.rdht(dhts.rdht(y, N, D, "fwd", theta), N, D, "inv", theta)
        np.testing.assert_allclose(restored, y, atol=5e-13)

    def test_fbt_is_involution(self):
        rng = np.random.default_rng(15)
        for N in range(1, 8):
            x = rng.normal(size=(3*(N+1), 4))
            np.testing.assert_allclose(dhts.fbt(dhts.fbt(x, N), N), x, atol=2e-13)

    def test_quadtree_block_roundtrip(self):
        x = np.random.default_rng(16).random((32, 32))
        coded, bits = dhts.dhtqt(x, 3)
        blocks = dhts.im2qtb(coded, bits, 3)
        np.testing.assert_array_equal(dhts.qtb2im(blocks, bits, x.shape, 3), coded)
        restored = dhts.idhtqt(coded, bits, 3)
        self.assertLess(float(np.mean((x-restored)**2)), 1e-4)

    def test_all_matlab_modules_have_python_counterpart(self):
        expected = {
            "angshow", "bincoef", "binpyr", "binpyr2", "bsmooth", "bsmooth2", "bt2dht", "chtmtx",
            "clssplot", "Contents", "ddht", "dht", "dht2", "dht2bt", "dht3", "dhtentr",
            "dhtgi", "dhti", "dhti2", "dhtJ", "dhtmorph", "dhtmtx", "dhtord", "dhtqt", "dhtshow",
            "dob2", "edht", "energy", "equaliz", "fbt", "fbt2", "fbt3", "fbtmtx", "gauge", "gbtmtx",
            "grafica", "graficaMapCoefs", "guidht", "hermite", "hermiteFiltersFreq", "idht", "idht2", "idht3",
            "idhtqt", "im2qtb", "imcorn", "imdht", "imdht2", "lorient", "matshow", "mddht2", "mdht",
            "mdht2", "mdhti", "mdhti2", "mgauge", "mrdht", "mrdht2", "mscode", "obtainOrdCoefs",
            "overshoot", "pdht", "qdht", "qtb2im", "qtplot", "rdht", "rdht2",
            "samplat", "sdht2", "xdht2", "zcross",
        }
        folder = Path(dhts.__file__).parent
        actual = {path.stem for path in folder.glob("*.py") if not path.stem.startswith("_")}
        self.assertTrue(expected.issubset(actual), expected-actual)

        # MATLAB scripts whose purpose is testing belong under dhts_py/tests.
        scripts = {path.stem for path in (folder / "tests").glob("test_*.py")}
        self.assertTrue(
            {"test_dht2_reconstruction", "test_prediction", "test_rotation"}.issubset(scripts)
        )


if __name__ == "__main__":
    unittest.main()
