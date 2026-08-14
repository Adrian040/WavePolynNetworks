"""Direct translation of ``edht.m``."""

import numpy as np


def edht(Y, k=3, *, return_bounds: bool = False):
    if isinstance(Y, (list, tuple)):
        result = [edht(value, k) for value in Y]
        return (result, None, None) if return_bounds else result
    arr = np.asarray(Y, dtype=np.float64)
    if arr.ndim >= 3 and arr.shape[2] > 1:
        result = np.stack([edht(arr[..., j], k) for j in range(arr.shape[2])], axis=2)
        return (result, None, None) if return_bounds else result
    bounds = np.atleast_1d(k).astype(float)
    if bounds.size > 1:
        ymin, ymax = float(bounds[0]), float(bounds[1])
    else:
        ymin = float(np.mean(arr) - bounds[0] * np.std(arr))
        ymax = float(np.mean(arr) + bounds[0] * np.std(arr))
    result = arr.copy() if ymin == ymax else np.clip((arr - ymin) / (ymax - ymin), 0, 1)
    return (result, ymin, ymax) if return_bounds else result


__all__ = ["edht"]
