"""Direct translation of ``hermite.m``."""

from math import factorial
from typing import Iterable, Sequence

import numpy as np


def hermite(D: int | Iterable[int], x: np.ndarray | Sequence[float] | None = None) -> np.ndarray:
    """Return the physicists' Hermite polynomials used by the toolbox."""

    degrees = np.atleast_1d(D).astype(int)
    if degrees.size == 0 or np.any(degrees < 0):
        raise ValueError("D must contain non-negative polynomial degrees")
    dmax = int(degrees.max())
    if x is None:
        result = np.zeros((dmax + 1, degrees.size), dtype=float)
    else:
        xx = np.asarray(x, dtype=float).ravel()
        result = np.zeros((xx.size, degrees.size), dtype=float)

    for column, n in enumerate(degrees):
        coeff = np.zeros(n + 1, dtype=float)
        for j in range(n // 2 + 1):
            power = n - 2 * j
            coeff[n - power] = (
                (-1.0) ** j
                * factorial(n)
                / (factorial(j) * factorial(power))
                * 2.0**power
            )
        if x is None:
            result[-(n + 1) :, column] = coeff
        else:
            result[:, column] = np.polyval(coeff, xx)
    return result


__all__ = ["hermite"]
