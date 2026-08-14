"""Direct translation of ``fbt.m``."""

import numpy as np

from ._core import integer, rbt_vector


def fbt(X, N: int, dim: int = 1, K0: float = np.sqrt(2.0)) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    axis = int(dim) - 1
    moved = np.moveaxis(arr, axis, 0)
    T = integer("N", N) + 1
    length = (moved.shape[0] // T) * T
    moved = moved[:length]
    flat = moved.reshape(length, -1)
    out = np.empty_like(flat)
    for start in range(0, length, T):
        out[start : start + T] = rbt_vector(flat[start : start + T], float(K0))
    return np.moveaxis(out.reshape((length,) + moved.shape[1:]), 0, axis)


__all__ = ["fbt"]
