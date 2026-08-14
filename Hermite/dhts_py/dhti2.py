"""Direct translation of ``dhti2.m``."""

from typing import Sequence

import numpy as np

from .dhtord import dhtord
from .idht2 import idht2


def dhti2(
    Y: np.ndarray,
    xsiz: Sequence[int],
    N: int,
    D: int,
    T: int,
    sumopt: str = "nosum",
    shape: str = "full",
    cod: str = " ",
) -> np.ndarray:
    """Interpolate individual 2-D coefficient contributions."""

    arr = np.asarray(Y, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[..., None]
    orders = dhtord(N, D, 2)
    extra = max(0, arr.shape[2] - len(orders))
    pieces = []
    for channel in range(arr.shape[2]):
        one = np.zeros(arr.shape[:2] + (len(orders),), dtype=np.float64)
        destination = 0 if channel <= extra else channel - extra
        one[..., destination] = arr[..., channel]
        pieces.append(idht2(one, xsiz, N, D, T, shape, cod))

    option = sumopt.lower()
    if option == "nosum":
        return np.stack(pieces, axis=-1)
    if option == "high":
        split = extra + 1
        return np.stack((np.sum(pieces[:split], axis=0), np.sum(pieces[split:], axis=0)), axis=-1)
    if option == "order":
        result = []
        for total in range(D + 1):
            selected = [
                pieces[i + extra]
                for i, order in enumerate(orders)
                if sum(order) == total and i + extra < len(pieces)
            ]
            result.append(np.sum(selected, axis=0) if selected else np.zeros(tuple(xsiz[:2])))
        return np.stack(result, axis=-1)
    if option == "predict":
        from .pdht import pdht

        return pdht(arr, tuple(xsiz[:2]), min(D, N), shape)
    raise ValueError(f"unrecognized sumopt {sumopt!r}")


__all__ = ["dhti2"]
