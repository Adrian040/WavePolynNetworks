"""Direct translation of ``gauge.m``."""

from math import comb

import numpy as np

from .dhtord import dhtord


def _binomial_norm(degree: int) -> np.ndarray:
    return np.sqrt(np.asarray([comb(degree, k) for k in range(degree + 1)], dtype=float))


def gauge(
    Y: np.ndarray,
    N: int,
    D: int,
    L: int | str = 1,
    *,
    components: bool = False,
    magnitude: bool = False,
):
    """Compute the general gauge condition from Cartesian DHT channels."""

    if isinstance(L, str):
        key = L.lower()
        L = 1 if key == "grad" else 2 if key == "hess" else int(key)
    L = int(L)
    if not 1 <= L <= D:
        raise ValueError("L must satisfy 1 <= L <= D")

    y = np.asarray(Y, dtype=np.float64)
    if y.ndim == 4:
        values = [
            gauge(y[..., k], N, D, L, components=components, magnitude=magnitude)
            for k in range(y.shape[3])
        ]
        if isinstance(values[0], tuple):
            return tuple(np.stack([value[i] for value in values], axis=-1) for i in range(len(values[0])))
        return np.stack(values, axis=-1)
    if y.ndim != 3:
        raise ValueError("gauge expects a coefficient stack")

    orders = dhtord(N, D, 2)
    selected = np.flatnonzero(orders.sum(axis=1) == L)
    if selected.size == 0:
        raise ValueError(f"coefficient order {L} is not available")
    degree = selected.size - 1
    block = y[..., selected]
    C = _binomial_norm(degree)
    Co = C[1::2] * (-1.0) ** np.arange(C[1::2].size)
    Ce = C[0::2] * (-1.0) ** np.arange(C[0::2].size)
    a = np.tensordot(block[..., 1::2], Co, axes=([-1], [0]))
    b = np.tensordot(block[..., 0::2], Ce, axes=([-1], [0]))

    if components:
        return a, b

    theta = np.arctan2(a, b) / max(degree, 1)
    if degree > 1:
        increments = np.arange(degree + 1) * np.pi / degree
        powers = np.arange(degree + 1)
        responses = np.stack(
            [
                np.sum(
                    block
                    * C.reshape((1,) * (block.ndim - 1) + (-1,))
                    * np.cos(theta[..., None] + increment) ** (degree - powers)
                    * np.sin(theta[..., None] + increment) ** powers,
                    axis=-1,
                )
                for increment in increments
            ],
            axis=-1,
        )
        theta = theta + np.argmax(np.abs(responses), axis=-1) * (np.pi / degree)
        if y.shape[-1] >= 3:
            sign = np.cos(theta) * y[..., 1] + np.sin(theta) * y[..., 2]
            theta = np.where(sign < 0, theta - np.pi, theta)
    return (theta, np.hypot(a, b)) if magnitude else theta


__all__ = ["gauge"]
