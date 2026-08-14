"""Direct translation of ``mdhti2.m``."""

from typing import Sequence

import numpy as np

from ._core import scale_degree
from .dhti2 import dhti2


def mdhti2(Y: Sequence[np.ndarray], xsiz: Sequence[int], D, sumopt: str = "nosum", shape: str = "full"):
    N0, degree = scale_degree(D, 8)
    T0 = int(round(np.sqrt(N0 / 2)))
    current = np.asarray(Y[-1], dtype=np.float64)
    for level in range(len(Y) - 2, -1, -1):
        saved = np.asarray(Y[level], dtype=np.float64)
        current = np.concatenate((dhti2(current, saved.shape[:2], 6, degree, 2, sumopt, shape, "l"), saved), axis=2)
    return dhti2(current, xsiz, N0, degree, T0, sumopt, shape, "h")


__all__ = ["mdhti2"]
