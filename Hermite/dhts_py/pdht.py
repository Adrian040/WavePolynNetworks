"""Direct translation of ``pdht.m``."""

from math import comb

import numpy as np

from .dhtord import dhtord
from .gauge import gauge
from .idht2 import idht2
from .rdht import rdht


def _predict_single(Y, ysiz, D: int, shape: str, cod: str = "", *args) -> np.ndarray:
    N, T, tau = 6, 2, 0.75
    degree = min(int(D), N)
    orders = dhtord(N, degree, 2)
    source = np.asarray(Y, dtype=np.float64)
    predictions = []
    for total in range(degree + 1):
        remaining = degree - total
        for j in range(total + 1):
            horizontal, vertical = total - j, j
            indices = np.flatnonzero((orders[:, 0] >= horizontal) & (orders[:, 1] >= vertical))
            z = source[..., indices].copy()
            position = 0
            for n in range(remaining + 1):
                for m in range(n + 1):
                    Cx = comb(n - m + horizontal, n - m)
                    Cy = comb(m + vertical, m)
                    z[..., position] *= np.sqrt(Cx * Cy * (1 - tau) ** total)
                    position += 1
            predictions.append(idht2(z, ysiz, N, remaining, T, shape, cod, *args))
    return np.stack(predictions, axis=-1)


def pdht(Y, D, tdir="fwd", cod: str = "", *args, shape: str | None = None):
    if not isinstance(Y, (list, tuple)):
        ysiz = D
        degree = int(tdir)
        mode = shape or (cod if len(cod) > 3 else "full")
        post = "" if len(cod) > 3 else cod
        return _predict_single(np.asarray(Y), ysiz, degree, mode, post, *args)
    mode = shape or (cod if len(cod) > 3 else "full")
    code = "" if len(cod) > 3 else cod
    result = [np.asarray(value, dtype=np.float64).copy() for value in Y]
    degree = min(int(D), 6)
    if str(tdir).lower() == "fwd":
        carrier = None
        for level in range(len(result) - 1, -1, -1):
            if level == len(result) - 1:
                full = result[level]
                carrier = full
            else:
                predicted = _predict_single(carrier, result[level].shape[:2], degree, mode)
                residual = result[level]
                count = min(residual.shape[2], predicted.shape[2] - 1)
                residual[..., :count] -= predicted[..., 1 : count + 1]
                full = np.concatenate((predicted[..., :1], residual), axis=2)
                carrier = full
                result[level] = full[..., 1:]
            if code.startswith("r"):
                angle = gauge(carrier, 6, degree, *(args or [1]))
                transformed = rdht(full, 6, degree, "fwd", angle)
                full = np.concatenate((transformed, angle[..., None] / (2 * np.pi)), axis=2)
                result[level] = full if level == len(result) - 1 else full[..., 1:]
        return result
    if str(tdir).lower() == "inv":
        carrier = None
        for level in range(len(result) - 1, -1, -1):
            full = result[level] if level == len(result) - 1 else np.concatenate((np.ones(result[level].shape[:2] + (1,)), result[level]), axis=2)
            if code.startswith("r"):
                angle, full = 2 * np.pi * full[..., -1], full[..., :-1]
                full = rdht(full, 6, degree, "inv", angle)
            if level < len(result) - 1:
                predicted = _predict_single(carrier, result[level].shape[:2], degree, mode)
                count = min(result[level].shape[2], predicted.shape[2] - 1)
                result[level][..., :count] += predicted[..., 1 : count + 1]
                carrier = np.concatenate((predicted[..., :1], result[level]), axis=2)
            else:
                carrier = full
                result[level] = full
        return result
    raise ValueError("tdir must be 'fwd' or 'inv'")


__all__ = ["pdht"]
