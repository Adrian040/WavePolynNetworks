"""Direct translation of ``rdht2.m``."""

import numpy as np

from .dhtord import dhtord
from .rdht import _rotate_block, _theta_map


def _rotate_tensor_pair(Y, N: int, D: int, axes: tuple[int, int], theta):
    orders = dhtord(N, D, 3)
    out = Y.copy()
    fixed_axis = ({0, 1, 2} - set(axes)).pop()
    for fixed in range(N + 1):
        indices_fixed = np.flatnonzero(orders[:, fixed_axis] == fixed)
        totals = np.unique(orders[indices_fixed][:, axes].sum(axis=1))
        for total in totals:
            indices = indices_fixed[orders[indices_fixed][:, axes].sum(axis=1) == total]
            indices = indices[np.argsort(orders[indices, axes[0]])]
            if indices.size > 1:
                out[..., indices] = _rotate_block(Y[..., indices], theta)
    return out


def rdht2(Y, N: int, D: int, flag: str = "fwd", theta=None, phi=None):
    arr = np.asarray(Y, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError("rdht2 expects (rows, columns, depth, coefficients)")
    if theta is None:
        theta = np.arctan2(arr[..., 2], arr[..., 1])
    if phi is None:
        phi = np.zeros(arr.shape[:3], dtype=float)
    theta = _theta_map(theta, arr.shape[:3])
    phi = _theta_map(phi, arr.shape[:3])
    if flag.lower() == "fwd":
        return _rotate_tensor_pair(_rotate_tensor_pair(arr, N, D, (0, 1), theta), N, D, (0, 2), phi)
    if flag.lower() == "inv":
        return _rotate_tensor_pair(_rotate_tensor_pair(arr, N, D, (0, 2), -phi), N, D, (0, 1), -theta)
    raise ValueError("flag must be 'fwd' or 'inv'")


__all__ = ["rdht2"]
