"""Direct translation of ``graficaMapCoefs.m``."""

import numpy as np

from .dhtord import dhtord


def _normalize01(X):
    arr = np.asarray(X, dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    return np.zeros_like(arr) if lo == hi else (arr - lo) / (hi - lo)


def graficaMapCoefs(Y, N: int, D: int, *, ax=None, show: bool = False) -> np.ndarray:
    arr = np.asarray(Y, dtype=np.float64)
    rows, cols = arr.shape[:2]
    side = min(N, D) + 1
    mosaic = np.ones((side * rows, side * cols), dtype=float)
    for channel, (horizontal, vertical) in enumerate(dhtord(N, D, 2)):
        if channel >= arr.shape[2]:
            break
        mosaic[vertical * rows : (vertical + 1) * rows, horizontal * cols : (horizontal + 1) * cols] = _normalize01(arr[..., channel])
    if ax is not None or show:
        import matplotlib.pyplot as plt

        target = ax or plt.subplots()[1]
        target.imshow(mosaic, cmap="gray")
        target.axis("off")
        if show:
            plt.show()
    return mosaic


__all__ = ["graficaMapCoefs"]
