"""Direct translation of ``obtainOrdCoefs.m``."""

import numpy as np

from .dhtord import dhtord


def obtainOrdCoefs(Y, N: int, D: int) -> dict[str, np.ndarray]:
    arr = np.asarray(Y)
    table = np.zeros((min(N, D) + 1, min(N, D) + 1), dtype=int)
    for channel, (horizontal, vertical) in enumerate(dhtord(N, D, 2), start=1):
        table[vertical, horizontal] = channel
    indices = table[table > 0] - 1
    return {"coefs": arr[..., indices], "mat": table}


__all__ = ["obtainOrdCoefs"]
