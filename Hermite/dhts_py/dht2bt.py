"""Direct translation of ``dht2bt.m``."""

from typing import Sequence

import numpy as np

from .dhtord import dhtord


def dht2bt(Z, siz: Sequence[int], N: int, D: int | None = None) -> np.ndarray:
    target = tuple(int(v) for v in siz[:2])
    T = int(N) + 1
    degree = N * N if D is None else int(D)
    if target[0] % T or target[1] % T:
        raise ValueError("siz dimensions must be multiples of N+1")
    out = np.zeros(target, dtype=np.float64)
    for channel, (horizontal, vertical) in enumerate(dhtord(N, degree, 2)):
        out[vertical::T, horizontal::T] = np.asarray(Z)[..., channel]
    weights = np.poly(np.ones(N))
    weights = np.sign(weights) * np.sqrt(np.abs(weights) / 2.0**N)
    return out / np.tile(np.outer(weights, weights), (target[0] // T, target[1] // T))


__all__ = ["dht2bt"]
