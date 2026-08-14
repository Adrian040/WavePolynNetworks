"""Direct translation of ``hermiteFiltersFreq.m``."""

import numpy as np

from .dht2 import dht2


def hermiteFiltersFreq(X, N: int, D: int, Tsub: int, rotate=None) -> np.ndarray:
    if np.isscalar(X) and float(X) == 0:
        image = np.zeros((N, N), dtype=float)
        image[0, 0] = 1
    else:
        image = np.asarray(X, dtype=np.float64)
    return dht2(image, N, D, Tsub, " " if rotate is None else rotate)


__all__ = ["hermiteFiltersFreq"]
