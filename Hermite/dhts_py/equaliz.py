"""Direct translation of ``equaliz.m``."""

from math import floor, log2

import numpy as np


def _histeq(image: np.ndarray, bins: int = 256) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if lo == hi:
        return np.zeros_like(arr)
    normalized = np.clip((arr - lo) / (hi - lo), 0, 1)
    hist, edges = np.histogram(normalized.ravel(), bins=bins, range=(0, 1))
    cdf = hist.cumsum().astype(float)
    nonzero = np.flatnonzero(cdf)
    if not nonzero.size or cdf[-1] == cdf[nonzero[0]]:
        return normalized
    cdf = (cdf - cdf[nonzero[0]]) / (cdf[-1] - cdf[nonzero[0]])
    return np.interp(normalized.ravel(), edges[:-1], cdf).reshape(arr.shape)


def equaliz(I) -> np.ndarray:
    from .imdht2 import imdht2
    from .mdht2 import mdht2

    image = np.asarray(I, dtype=np.float64)
    levels = max(1, floor(log2(min(image.shape[:2])) / 2))
    coefficients = mdht2(_histeq(image), 4, levels, "symm")
    coefficients[-1][..., 0] = 0
    reconstructed = imdht2(coefficients, image.shape, 4, "symm")
    lo, hi = np.nanmin(reconstructed), np.nanmax(reconstructed)
    return np.zeros_like(reconstructed) if lo == hi else (reconstructed - lo) / (hi - lo)


__all__ = ["equaliz"]
