"""Direct translation of ``lorient.m``."""

from math import floor, log2

import numpy as np

from .mdht2 import mdht2
from .mdhti2 import mdhti2


def lorient(X, M: int | None = None):
    image = np.asarray(X, dtype=np.float64)
    levels = floor(log2(min(image.shape)) / 2) if M is None else int(M)
    coeff = mdhti2(mdht2(image, 1, levels), image.shape, 0)
    xx_ch = coeff[..., 1::2]
    yy_ch = coeff[..., 2::2]
    count = min(xx_ch.shape[-1], yy_ch.shape[-1])
    yy = np.sum(2 * xx_ch[..., :count] * yy_ch[..., :count], axis=-1)
    xx = np.sum(xx_ch[..., :count] ** 2 - yy_ch[..., :count] ** 2, axis=-1)
    return 0.5 * np.arctan2(yy, xx) + np.pi / 2, np.hypot(xx, yy)


__all__ = ["lorient"]
