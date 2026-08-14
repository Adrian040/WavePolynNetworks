"""Direct translation of ``fbtmtx.m``."""

import numpy as np

from .dhtmtx import dhtmtx


def fbtmtx(N: int) -> np.ndarray:
    B = dhtmtx(N)
    scale = (2.0**N) / np.sqrt((2.0**N) * B[:, 0])
    return np.rint(B * scale[None, :])


__all__ = ["fbtmtx"]
