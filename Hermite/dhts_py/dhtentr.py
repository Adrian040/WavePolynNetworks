"""Direct translation of ``dhtentr.m``."""

import numpy as np


def dhtentr(Y, L: int = 255):
    if isinstance(Y, (list, tuple)):
        return [dhtentr(value, L) for value in Y]
    arr = np.asarray(Y, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[..., None]
    values = []
    for channel in range(arr.shape[2]):
        shifted = arr[..., channel] + (0.5 if channel > 0 else 0.0)
        counts, _ = np.histogram(np.clip(shifted, 0, 1), bins=int(L), range=(0, 1))
        probability = counts[counts > 0] / counts.sum()
        values.append(float(-np.sum(probability * np.log2(probability))))
    return np.asarray(values)


__all__ = ["dhtentr"]
