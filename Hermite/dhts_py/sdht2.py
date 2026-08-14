"""Direct translation of ``sdht2.m`` with documented external-data fallback."""

import numpy as np

from .dhtord import dhtord
from .energy import energy
from .rdht import _theta_map, rdht


_MISSING = object()


def _perceptual_tau(rotated, N: int, D: int) -> np.ndarray:
    total = energy(rotated, N, D, "ac")
    one_d = energy(rotated, N, D, "1d")
    ratio = one_d / (total + np.finfo(float).eps)
    active = total > np.nanmedian(total)
    label_1d = active & (ratio >= 0.75)
    label_2d = active & ~label_1d
    return np.stack((label_1d | label_2d, label_2d), axis=-1).astype(float)


def sdht2(Y, N: int, D: int, tau=None, theta=_MISSING):
    arr = np.asarray(Y, dtype=np.float64).copy()
    if arr.ndim == 4:
        bands = [sdht2(arr[..., k], N, D, tau, theta) for k in range(arr.shape[3])]
        return np.stack([band[0] for band in bands], axis=-1), bands[0][1], bands[0][2]
    if theta is _MISSING:
        if tau is None:
            first = arr[..., 1] ** 2 + arr[..., 2] ** 2
            std = np.std(first)
            tau = np.full(first.shape, 0.5) if std == 0 else (1 + np.tanh(first / std - 1)) / 2
        elif callable(tau):
            tau = tau(arr)
        tau_map = np.asarray(tau, dtype=float)
        for channel, order in enumerate(dhtord(N, D, 2)):
            if channel:
                arr[..., channel] *= tau_map ** int(order.sum())
        return arr, tau_map, None

    if theta is None:
        angle = np.arctan2(arr[..., 2], arr[..., 1]) if tau is None else np.arctan2(arr[..., 1], -arr[..., 2])
    elif callable(theta):
        angle = theta(arr)
    else:
        angle = _theta_map(theta, arr.shape[:2])
    rotated = rdht(arr, N, D, "fwd", angle)
    tau_map = _perceptual_tau(rotated, N, D) if tau is None else np.asarray(tau, dtype=float)
    if tau_map.ndim == 2:
        tau_map = tau_map[..., None]
    for channel, (horizontal, vertical) in enumerate(dhtord(N, D, 2)):
        if channel:
            factor = tau_map[..., 0] ** horizontal
            if tau_map.shape[-1] > 1:
                factor *= tau_map[..., 1] ** vertical
            rotated[..., channel] *= factor
    return rdht(rotated, N, D, "inv", angle), tau_map, angle


__all__ = ["sdht2"]
