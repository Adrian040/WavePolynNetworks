"""Direct translation of ``bincoef.m``."""

from math import comb

import numpy as np


def bincoef(n, r) -> np.ndarray:
    nn, rr = np.asarray(n, dtype=int), np.asarray(r, dtype=int)
    if np.any(nn < 0) or np.any(rr < 0):
        raise ValueError("n and r must be non-negative")
    if nn.size == rr.size:
        values = np.asarray([comb(int(a), int(b)) if b <= a else 0 for a, b in zip(nn.ravel(), rr.ravel())])
        return values.reshape(nn.shape)
    return np.asarray([[comb(int(a), int(b)) if b <= a else 0 for b in rr.ravel()] for a in nn.ravel()])


__all__ = ["bincoef"]
