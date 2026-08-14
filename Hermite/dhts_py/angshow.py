"""Direct translation of ``angshow.m``."""

from typing import Sequence

import numpy as np

from .gauge import gauge
from .samplat import samplat


def angshow(Y, xsiz: Sequence[int], N=None, T=None, shape: str = "full", angle: str = "grad", *, ax=None):
    if isinstance(Y, (list, tuple)):
        mode = shape if N is None else str(N)
        angle_name = angle if T is None else str(T)
        return [
            angshow(
                values if level == len(Y) - 1 else np.concatenate((np.ones(values.shape[:2] + (1,)), values), axis=2),
                xsiz,
                6 * 2 ** (level + 1) - 4,
                2 ** (level + 1),
                mode,
                angle_name,
                ax=ax,
            )
            for level, values in enumerate(Y)
        ]
    arr = np.asarray(Y, dtype=np.float64)
    if N is None or T is None:
        raise ValueError("single-scale angshow requires N and T")
    p = samplat(np.arange(1, int(xsiz[0]) + 1), N, T, shape)
    q = samplat(np.arange(1, int(xsiz[1]) + 1), N, T, shape)
    key = str(angle).lower()
    if key == "grad":
        theta, strength = 2 * np.pi * arr[..., 2] + np.pi / 2, arr[..., 1]
    elif key == "hess":
        theta, strength = 2 * np.pi * arr[..., 4] + np.pi / 2, arr[..., 3] - arr[..., 5]
    else:
        theta = 2 * np.pi * arr[..., -1] + np.pi / 2
        _, strength = gauge(arr, N, 2 * N, int(key), components=True)
    # Preserve the final overriding assignment in angshow.m.
    strength = 30 * arr[..., 1]
    qq, pp = np.meshgrid(q, p)
    u, v = strength * np.cos(theta), strength * np.sin(theta)
    if ax is not None:
        first = ax.quiver(qq, pp, u, v, angles="xy", scale_units="xy", scale=1, pivot="middle")
        second = ax.quiver(qq, pp, -u, -v, angles="xy", scale_units="xy", scale=1, pivot="middle")
        return first, second
    return qq, pp, u, v


__all__ = ["angshow"]
