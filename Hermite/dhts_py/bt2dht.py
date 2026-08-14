"""Direct translation of ``bt2dht.m``."""

import numpy as np

from .dhtord import dhtord


def bt2dht(Y, N: int, D: int | None = None) -> np.ndarray:
    arr = np.asarray(Y, dtype=np.float64).copy()
    T = int(N) + 1
    degree = N * N if D is None else int(D)
    if arr.shape[0] % T or arr.shape[1] % T:
        raise ValueError("both dimensions of Y must be multiples of N+1")
    weights = np.poly(np.ones(N))
    weights = np.sign(weights) * np.sqrt(np.abs(weights) / 2.0**N)
    arr *= np.tile(np.outer(weights, weights), (arr.shape[0] // T, arr.shape[1] // T))
    return np.stack([arr[vertical::T, horizontal::T] for horizontal, vertical in dhtord(N, degree, 2)], axis=-1)


__all__ = ["bt2dht"]
