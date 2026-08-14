"""Direct translation of ``dhtmorph.m``."""

import numpy as np
from scipy.signal import convolve2d

from .dhtord import dhtord
from .gauge import gauge
from .rdht import _binomial_norm, rdht


def _dhtshift(Y, N: int, D: int, M: int, theta):
    rotated = rdht(Y, N, D, "fwd", theta)
    flat = rotated.reshape(-1, rotated.shape[-1])
    out = np.zeros_like(flat)
    orders = dhtord(N, D, 2)
    norms = _binomial_norm(N)
    flat = flat / norms[orders[:, 0]][None, :]
    kernel = (-1) ** M * np.poly(np.ones(M))
    reduced = np.r_[_binomial_norm(N - M), np.zeros(M)]
    for second in range(min(D, N) + 1):
        indices = np.flatnonzero(orders[:, 1] == second)
        imax = min(min(N - M, D - M) + 1, indices.size)
        source_count = min(indices.size, imax + M)
        source = np.c_[flat[:, indices[:source_count]], np.zeros((flat.shape[0], imax + M - source_count))]
        if imax:
            out[:, indices[:imax]] = convolve2d(source, kernel[None, :], mode="valid")[:, :imax] * reduced[:imax]
    return rdht(out.reshape(rotated.shape), N, D, "inv", theta)


def dhtmorph(Y, N: int, D: int, oper: str, M: int | None = None, theta=None):
    amount = int(N / 2 if M is None or M < 1 else M)
    angle = gauge(Y, N, D, "grad") if theta is None else (gauge(Y, N, D, theta) if isinstance(theta, str) else theta)
    key = oper.lower()
    if key == "erode":
        result = _dhtshift(Y, N, D, amount, angle)
    elif key == "dilate":
        result = _dhtshift(Y, N, D, amount, angle + np.pi)
    elif key == "close":
        result = _dhtshift(_dhtshift(Y, N, D, amount, angle + np.pi), N, D, amount, angle)
    elif key == "open":
        result = _dhtshift(_dhtshift(Y, N, D, amount, angle), N, D, amount, angle + np.pi)
    else:
        raise ValueError("oper must be erode, dilate, close, or open")
    return result, angle


__all__ = ["dhtmorph"]
