"""Direct translation of ``dht3.m``."""

import numpy as np
from scipy.signal import convolve

from ._core import pad_spatial
from .dhtmtx import dhtmtx
from .dhtord import dhtord


def dht3(X, N: int, D: int, T: int, cod: str = "", *args, shape: str | None = None):
    mode = shape or (cod if len(cod) > 2 else "full")
    code = "" if len(cod) > 2 else cod
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError("dht3 expects a 3-D array")
    H = dhtmtx(N, min(D, N))
    work, conv_mode = X, mode
    if mode in {"symm", "repeat", "asymm", "cyclic"}:
        work, conv_mode = pad_spatial(X, N, mode, 3), "valid"
    channels = []
    for x_order, y_order, z_order in dhtord(N, D, 3):
        kernel = H[:, y_order, None, None] * H[None, :, x_order, None] * H[None, None, :, z_order]
        channels.append(convolve(work, kernel, mode=conv_mode)[::T, ::T, ::T])
    Y = np.stack(channels, axis=-1)
    if code.lower().startswith("r"):
        from .rdht2 import rdht2

        Y = rdht2(Y, N, D, "fwd", *args)
    return Y


__all__ = ["dht3"]
