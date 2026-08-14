"""Direct translation of ``ddht.m``."""

import numpy as np

from .gauge import gauge
from .rdht import _binomial_norm, _channel_blocks, _theta_map


def ddht(Y, N: int, D: int, tdir: str = "fwd", theta=None, *, return_theta: bool = False):
    arr = np.asarray(Y, dtype=np.float64)
    if arr.ndim == 4:
        values = [ddht(arr[..., k], N, D, tdir, theta, return_theta=return_theta) for k in range(arr.shape[3])]
        if return_theta:
            return np.stack([value[0] for value in values], axis=-1), values[0][1]
        return np.stack(values, axis=-1)
    condition = "grad" if theta is None else (theta.lower() if isinstance(theta, str) else "none")
    work = arr.copy()
    null_channel = None
    if tdir.lower() == "fwd":
        if condition == "grad":
            angle, null_channel = gauge(work, N, D, 1), 2
        elif condition == "hess":
            angle, null_channel = gauge(work, N, D, 2), 4
        elif condition == "none":
            angle = _theta_map(theta, work.shape[:2])
        else:
            angle = gauge(work, N, D, int(condition))
            if not return_theta:
                null_channel = work.shape[-1]
                work = np.concatenate((work, np.zeros(work.shape[:2] + (1,))), axis=-1)
        out = work.copy()
        for start, degree in _channel_blocks(N, D):
            C = _binomial_norm(degree)
            powers = np.arange(degree + 1)
            block = work[..., start : start + degree + 1]
            for j in range(degree + 1):
                direction = angle + j * np.pi / (degree + 1)
                out[..., start + j] = np.sum(
                    block
                    * C
                    * np.cos(direction)[..., None] ** (degree - powers)
                    * np.sin(direction)[..., None] ** powers,
                    axis=-1,
                )
        if null_channel is not None:
            out[..., null_channel] = angle / (2 * np.pi)
    elif tdir.lower() == "inv":
        if condition == "grad":
            angle = 2 * np.pi * work[..., 2]
            work[..., 2] = 0
        elif condition == "hess":
            angle = 2 * np.pi * work[..., 4]
            work[..., 4] = 0
        elif condition == "none":
            angle = _theta_map(theta, work.shape[:2])
        else:
            angle = 2 * np.pi * work[..., -1]
            work = work[..., :-1]
        out = work.copy()
        for start, degree in _channel_blocks(N, D):
            C = _binomial_norm(degree)
            powers = np.arange(degree + 1)
            directions = angle[..., None] + np.arange(degree + 1) * np.pi / (degree + 1)
            basis = (
                np.cos(directions)[..., None] ** (degree - powers)
                * np.sin(directions)[..., None] ** powers
            )
            norm = np.sum(np.sin(np.arange(degree + 1) * np.pi / (degree + 1)) ** (2 * degree))
            for m in range(degree + 1):
                out[..., start + m] = (
                    C[m]
                    * np.sum(work[..., start : start + degree + 1] * basis[..., m], axis=-1)
                    / norm
                )
    else:
        raise ValueError("tdir must be 'fwd' or 'inv'")
    return (out, angle) if return_theta else out


__all__ = ["ddht"]
