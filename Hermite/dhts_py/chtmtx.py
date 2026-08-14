"""Direct translation of ``chtmtx.m``."""

from math import factorial, floor
from typing import Sequence

import numpy as np

from ._core import integer
from .hermite import hermite


def chtmtx(s: float, D: int, x: np.ndarray | Sequence[float] | None = None) -> np.ndarray:
    if s <= 0:
        raise ValueError("s must be positive")
    D = integer("D", D)
    xx = (-4 * s + np.arange(floor(8 * s) + 1)) if x is None else np.asarray(x, dtype=float).ravel()
    n = np.arange(D + 1)
    c = (-1.0) ** n * np.sqrt(2.0**n * np.array([factorial(int(v)) for v in n]))
    gaussian = np.exp(-(xx**2) / (4 * s)) / np.sqrt(4 * s * np.pi)
    return hermite(n, xx / np.sqrt(4 * s)) * gaussian[:, None] / c[None, :]


__all__ = ["chtmtx"]
