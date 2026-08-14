"""Direct translation of ``dhtord.m``."""

import numpy as np

from ._core import integer


def dhtord(N: int, D: int, K: int = 2) -> np.ndarray:
    """Return orders in the exact channel sequence produced by MATLAB."""

    N = integer("N", N)
    D = integer("D", D)
    if K == 1:
        return np.arange(min(D, N) + 1, dtype=int)[:, None]
    if K == 2:
        return np.asarray(
            [
                (total - m, m)
                for total in range(min(D, 2 * N) + 1)
                for m in range(max(0, total - N), min(N, total) + 1)
            ],
            dtype=int,
        )
    if K == 3:
        return np.asarray(
            [
                (b, a, c)
                for c in range(min(D, N) + 1)
                for total in range(min(D - c, 2 * N) + 1)
                for a in range(max(0, total - N), min(N, total) + 1)
                for b in [total - a]
            ],
            dtype=int,
        )
    raise ValueError("K must be 1, 2, or 3")


__all__ = ["dhtord"]
