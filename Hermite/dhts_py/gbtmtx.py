"""Translation of ``gbtmtx.m`` (whose MATLAB function name is a typo)."""

from math import floor

import numpy as np

from ._core import convmtx, integer


def gbtmtx(N: int, theta: float, D: int | None = None, T: int | None = None):
    N = integer("N", N)
    D = N if D is None else min(N, integer("D", D))
    if N == 0:
        H = np.ones((1, 1), dtype=float)
        return (H, H.copy()) if T is not None else H
    B = np.array(
        [[np.sin(theta), np.cos(theta)], [np.cos(theta), -np.sin(theta)]],
        dtype=float,
    )
    H = B.copy() if D > 0 else B[:, :1].copy()
    for _m in range(2, N + 1):
        n = H.shape[1]
        smoothed = np.column_stack([np.convolve(B[:, 0], H[:, j]) for j in range(n)])
        H = np.column_stack((smoothed, np.convolve(B[:, 1], H[:, -1]))) if n <= D else smoothed
    if T is None:
        return H
    T = integer("T", T, 1)
    if T > N:
        raise ValueError("T must satisfy 1 <= T <= N")
    W = convmtx(H[:, 0], N + 1)
    first = N - floor(N / T) * T
    weights = W[np.arange(first, 2 * N + 1, T), :].sum(axis=0)
    return H, H[::-1, :] / weights[:, None]


__all__ = ["gbtmtx"]
