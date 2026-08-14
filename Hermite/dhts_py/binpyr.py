"""Direct translation of ``binpyr.m``."""

import numpy as np

from .dht import dht


def binpyr(X, M, dim: int | None = None, shape: str = "full") -> list[np.ndarray]:
    values = np.atleast_1d(M).astype(int)
    levels = int(values[0])
    N = int(values[1]) if values.size > 1 else 8
    T = int(round(np.sqrt(N / 2)))
    result = [dht(X, N, 0, T, dim, shape)]
    for _ in range(levels):
        result.append(2 * dht(np.squeeze(result[-1], axis=-1), 6, 0, 2, dim, shape))
    return result


__all__ = ["binpyr"]
