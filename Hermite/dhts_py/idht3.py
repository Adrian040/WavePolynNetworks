"""Direct translation of ``idht3.m``."""

from typing import Sequence

import numpy as np
from scipy.signal import convolve

from ._core import synthesis_filters
from .dhtord import dhtord


def idht3(Y, xsiz: Sequence[int], N: int, D: int, T: int, cod: str = "", *args, shape: str | None = None):
    mode = shape or (cod if len(cod) > 2 else "full")
    code = "" if len(cod) > 2 else cod
    arr = np.asarray(Y, dtype=np.float64)
    if code.lower().startswith("r"):
        from .rdht2 import rdht2

        arr = rdht2(arr, N, D, "inv", *args)
    target = tuple(int(v) for v in xsiz[:3])
    if mode == "full":
        ysiz, conv_mode = tuple(v + N for v in target), "valid"
    elif mode == "same":
        ysiz, conv_mode = target, "same"
    elif mode == "valid":
        ysiz, conv_mode = tuple(v - N for v in target), "full"
    else:
        raise NotImplementedError("3-D inverse boundary extension supports full/same/valid")
    G = synthesis_filters(N, min(D, N), T)
    lattice = tuple(np.arange(0, v, T) for v in ysiz)
    expected = tuple(len(v) for v in lattice)
    if arr.shape[:3] != expected:
        raise ValueError(f"expected coefficient grid {expected}, got {arr.shape[:3]}")
    X = np.zeros(target, dtype=np.float64)
    for channel, (x_order, y_order, z_order) in enumerate(dhtord(N, D, 3)):
        yi = np.zeros(ysiz, dtype=np.float64)
        yi[np.ix_(*lattice)] = arr[..., channel]
        kernel = G[:, y_order, None, None] * G[None, :, x_order, None] * G[None, None, :, z_order]
        X += convolve(yi, kernel, mode=conv_mode)
    return X


__all__ = ["idht3"]
