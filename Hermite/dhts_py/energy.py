"""Direct translation of ``energy.m``."""

from typing import Sequence

import numpy as np

from .dhtord import dhtord


def energy(Y, N: int, D: int | str | None = None, cod: str = "ac"):
    if isinstance(Y, (list, tuple)):
        code = D if isinstance(D, str) else cod
        if isinstance(N, Sequence):
            N0, degree = int(N[0]), int(N[1])
        else:
            N0, degree = 8, int(N)
        result = []
        for level, values in enumerate(Y):
            scale_n = N0 if level == 0 else 6
            stack = values if level == len(Y) - 1 else np.concatenate((np.ones(values.shape[:2] + (1,)), values), axis=2)
            result.append(energy(stack, scale_n, degree, str(code)))
        return result
    if D is None or isinstance(D, str):
        raise ValueError("single-scale energy requires D")
    arr = np.asarray(Y, dtype=np.float64)
    orders = dhtord(N, int(D), 2)
    key = cod.lower()
    if key == "dc":
        mask = np.ones(len(orders), bool)
    elif key == "ac":
        mask = np.arange(len(orders)) > 0
    elif key == "esym":
        mask = (orders[:, 0] % 2 == 0) & (np.arange(len(orders)) > 0)
    elif key == "osym":
        mask = orders[:, 0] % 2 == 1
    elif key == "1d":
        mask = (orders[:, 1] == 0) & (np.arange(len(orders)) > 0)
    elif key == "e1d":
        mask = (orders[:, 1] == 0) & (orders[:, 0] % 2 == 0) & (np.arange(len(orders)) > 0)
    elif key == "o1d":
        mask = (orders[:, 1] == 0) & (orders[:, 0] % 2 == 1)
    elif key == "2d":
        mask = orders[:, 1] > 0
    else:
        raise ValueError(f"unknown energy code {cod!r}")
    return np.sqrt(np.sum(arr[..., : len(orders)][..., mask] ** 2, axis=-1))


__all__ = ["energy"]
