"""Direct translation of ``binpyr2.m``."""

import numpy as np

from .dht2 import dht2


def binpyr2(X, M, shape: str = "full") -> list[np.ndarray]:
    values = np.atleast_1d(M).astype(int)
    levels = int(values[0])
    N = int(values[1]) if values.size > 1 else 8
    result = [np.squeeze(dht2(X, N, 0, int(round(np.sqrt(N / 2))), shape), axis=2)]
    for _ in range(1, levels):
        result.append(np.squeeze(dht2(result[-1], 6, 0, 2, shape), axis=2))
    return result


__all__ = ["binpyr2"]
