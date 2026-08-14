"""Direct translation of ``mdht.m``."""

import numpy as np

from ._core import scale_degree
from .dht import dht


def mdht(X, D, M: int, dim: int | None = None, cod: str = " ", *args, shape: str | None = None):
    if shape is None and len(cod) > 3:
        shape, cod = cod.lower(), (str(args[0]) if args else " ")
        args = args[1:] if args else ()
    shape = shape or "full"
    N, degree = scale_degree(D, 8)
    T = int(round(np.sqrt(N / 2)))
    current = np.asarray(X, dtype=np.float64)
    result = []
    for level in range(int(M)):
        prefix = "h" if level == 0 else "l"
        if level == M - 1:
            result.append(dht(current, N, degree, T, dim, shape, prefix + cod, *args))
        else:
            high, current = dht(current, N, degree, T, dim, shape, prefix + cod, *args, return_lowpass=True)
            result.append(high)
        N, T = 6, 2
    return result


__all__ = ["mdht"]
