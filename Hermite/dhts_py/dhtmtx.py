"""Direct translation of ``dhtmtx.m``."""

from math import floor

import numpy as np

from ._core import convmtx, integer


def dhtmtx(N: int, D: int | None = None, T: int | None = None):
    """Construct the binomial/Krawtchouk analysis and synthesis filters."""

    N = integer("N", N)
    D = N if D is None else min(N, integer("D", D))
    if N == 0:
        H = np.ones((1, 1), dtype=np.float64)
        return (H, H.copy()) if T is not None else H

    B = np.array([[0.5, 0.5], [0.5, -0.5]], dtype=np.float64)
    H = B.copy() if D > 0 else B[:, :1].copy()

    for _m in range(2, N + 1):
        n = H.shape[1]
        smoothed = np.column_stack(
            [np.convolve(B[:, 0], H[:, column], mode="full") for column in range(n)]
        )
        if n <= D:
            H = np.column_stack((smoothed, np.convolve(B[:, 1], H[:, -1], mode="full")))
        else:
            H = smoothed

    C = 2.0 ** (N / 2.0) * np.sqrt(H[: D + 1, 0])
    H *= C[None, :]

    if T is None:
        return H

    T = integer("T", T, 1)
    if T > N:
        raise ValueError("T must satisfy 1 <= T <= N")
    W = convmtx(H[:, 0], N + 1)
    first = N - floor(N / T) * T
    W = W[np.arange(first, 2 * N + 1, T), :].sum(axis=0)
    if np.any(W == 0):
        raise ZeroDivisionError("zero interpolation weight in dhtmtx")
    G = H[::-1, :] / W[:, None]
    return H, G


__all__ = ["dhtmtx"]
