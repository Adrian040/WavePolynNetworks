"""Python version of rot_test.m."""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import rotate

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dhts_py.dht2 import dht2


def run_rot_test(N: int = 8, angle_degrees: float = 30.0):
    D, T = 2 * N, 1
    X = np.zeros((2 * N + 1, 2 * N + 1), dtype=float)
    X[N:, :] = 1
    X[N, :] = 0.5
    X = rotate(X, angle_degrees, reshape=False, order=1, mode="constant")
    X = X[N // 2 : -N // 2, N // 2 : -N // 2]
    coefficients, theta = dht2(X, N, D, T, "valid", "r", "1", return_aux=True)
    return coefficients, theta


if __name__ == "__main__":
    coefs, theta = run_rot_test()
    print("coefficients:", coefs.shape, "mean theta (deg):", np.degrees(np.mean(theta)))
