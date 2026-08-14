"""Direct translation of ``dob2.m``."""

import numpy as np
from scipy.signal import convolve2d


def dob2(X, Nc: int, Ns: int | None = None) -> np.ndarray:
    central = 2 * int(Nc)
    surround = (4 * central if central > 0 else 4) if Ns is None else int(Ns)
    bc = np.poly(-np.ones(central)) / 2.0**central
    bs = np.poly(-np.ones(surround)) / 2.0**surround
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 2:
        return convolve2d(arr, np.outer(bc, bc), mode="same") - convolve2d(arr, np.outer(bs, bs), mode="same")
    return np.stack([dob2(arr[..., k], Nc, surround) for k in range(arr.shape[2])], axis=-1)


__all__ = ["dob2"]
