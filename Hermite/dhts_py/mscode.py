"""Direct translation of ``mscode.m``."""

from math import floor, log2

import numpy as np

from .bsmooth2 import bsmooth2


def mscode(X, S=None, Tol: float = 0, *args):
    image = np.asarray(X, dtype=np.float64)
    if S is None:
        levels = min(floor(log2(value)) for value in image.shape[:2])
        scales = 4.0 ** np.arange(1, levels + 1)
    else:
        scales = np.atleast_1d(S)
    smooth = bsmooth2(image + Tol, scales, *args)
    code = np.zeros(image.shape, dtype=np.uint64)
    transition = np.ones(image.shape, dtype=int)
    previous = np.ones(image.shape, dtype=bool)
    for n in range(len(scales) - 1, -1, -1):
        bit = image >= smooth[..., n]
        code |= bit.astype(np.uint64) << n
        transition[(transition == 1) & previous & ~bit] = n + 2
        previous = bit
    transition[transition == len(scales) + 1] = 0
    stack = np.concatenate((image[..., None], smooth), axis=-1)
    selected = np.take_along_axis(stack, transition[..., None], axis=-1)[..., 0]
    return code, transition, selected


__all__ = ["mscode"]
