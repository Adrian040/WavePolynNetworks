"""Direct translation of ``rdht.m``."""

from math import comb

import numpy as np

from .gauge import gauge


def _binomial_norm(degree: int) -> np.ndarray:
    return np.sqrt(np.asarray([comb(degree, k) for k in range(degree + 1)], dtype=float))


def _theta_map(theta, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(theta, dtype=float)
    if value.ndim == 0:
        return np.full(shape, float(value), dtype=float)
    if value.shape != shape:
        try:
            return np.broadcast_to(value, shape).astype(float, copy=False)
        except ValueError as exc:
            raise ValueError(f"theta has shape {value.shape}; expected {shape}") from exc
    return value


def _channel_blocks(N: int, D: int) -> list[tuple[int, int]]:
    blocks = []
    start = 1
    for total in range(1, min(D, N) + 1):
        blocks.append((start, total))
        start += total + 1
    for offset in range(1, min(D - N, N - 1) + 1):
        degree = N - offset
        blocks.append((start, degree))
        start += degree + 1
    return blocks


def _rotate_block(block: np.ndarray, theta: np.ndarray) -> np.ndarray:
    degree = block.shape[-1] - 1
    if degree <= 0:
        return block.copy()
    c, s = np.cos(theta), np.sin(theta)
    C = _binomial_norm(degree)
    h = np.asarray(block, dtype=np.float64).copy()
    if degree > 1:
        h[..., 1:degree] /= C[1:degree]
    out = np.empty_like(h)
    hlen = degree + 1
    for m in range(degree):
        low = h.copy()
        llen = hlen
        for _j in range(m, degree):
            low = c[..., None] * low[..., : llen - 1] + s[..., None] * low[..., 1:llen]
            llen -= 1
        out[..., m] = low[..., 0] * C[m]
        h = c[..., None] * h[..., 1:hlen] - s[..., None] * h[..., : hlen - 1]
        hlen -= 1
    out[..., degree] = h[..., 0]
    return out


def rdht(
    Y: np.ndarray,
    N: int,
    D: int,
    tdir: str = "fwd",
    theta=None,
    *,
    return_theta: bool = False,
):
    """Rotate normalized Cartesian DHT coefficient blocks."""

    y = np.asarray(Y, dtype=np.float64)
    if y.ndim == 4:
        values = [rdht(y[..., k], N, D, tdir, theta, return_theta=return_theta) for k in range(y.shape[3])]
        if return_theta:
            return np.stack([value[0] for value in values], axis=-1), values[0][1]
        return np.stack(values, axis=-1)
    if y.ndim < 3:
        return (y.copy(), theta) if return_theta else y.copy()

    direction = tdir.lower()
    gaucond = "grad" if theta is None else (theta.lower() if isinstance(theta, str) else "none")
    work = y.copy()
    null_channel = None

    if direction == "inv":
        if gaucond in {"grad", "grad180"}:
            angle = -2 * np.pi * work[..., 2]
            work[..., 2] = 0
        elif gaucond == "hess":
            angle = -2 * np.pi * work[..., 4]
            work[..., 4] = 0
        elif gaucond == "none":
            angle = -_theta_map(theta, work.shape[:2])
        else:
            angle = -2 * np.pi * work[..., -1]
            work = work[..., :-1]
    elif direction == "fwd":
        if gaucond == "grad":
            angle, null_channel = gauge(work, N, D, 1), 2
        elif gaucond == "grad180":
            angle, null_channel = gauge(work, N, D, 1) + np.pi, 2
        elif gaucond == "hess":
            angle, null_channel = gauge(work, N, D, 2), 4
        elif gaucond == "none":
            angle = _theta_map(theta, work.shape[:2])
        else:
            angle = gauge(work, N, D, int(gaucond))
            if not return_theta:
                null_channel = work.shape[-1]
                work = np.concatenate((work, np.zeros(work.shape[:2] + (1,))), axis=-1)
    else:
        raise ValueError("tdir must be 'fwd' or 'inv'")

    z = work.copy()
    for start, degree in _channel_blocks(N, D):
        stop = start + degree + 1
        if stop <= work.shape[-1]:
            z[..., start:stop] = _rotate_block(work[..., start:stop], angle)
    if null_channel is not None:
        z[..., null_channel] = angle / (2 * np.pi)
    return (z, angle if direction == "fwd" else -angle) if return_theta else z


__all__ = ["rdht"]
