"""Direct translation of ``xdht2.m``."""

import numpy as np
from scipy.signal import convolve2d

from .dhtord import dhtord
from .rdht import _binomial_norm, rdht


def xdht2(Y, N: int, D: int, M, msk=None, theta=None, *args):
    arr = np.asarray(Y, dtype=np.float64)
    rotated = rdht(arr, N, D, "fwd", theta) if theta is not None else arr.copy()
    values = np.atleast_1d(M).astype(int)
    shift = min(D, int(values[0]))
    nmax = int(values[1]) if values.size > 1 else N - shift
    mmax = int(values[2]) if values.size > 2 else N
    order = min(int(values[3]), 1) if values.size > 3 else 0
    nmax = min(nmax, D - shift) - order
    mmax = min(mmax, D)
    if nmax < 0:
        raise ValueError("increase D or decrease M")
    flat = rotated.reshape(-1, rotated.shape[-1])
    out = flat.copy()
    if msk is None:
        mask = np.ones(flat.shape[0], bool)
    elif np.isscalar(msk):
        mask = flat[:, 1] > float(msk)
    else:
        mask = np.asarray(msk, dtype=bool).ravel()
    orders = dhtord(N, D, 2)
    norms = _binomial_norm(N)
    work = flat.copy()
    work[mask, :] /= norms[orders[:, 0]][None, :]
    kernel = (-1) ** shift * np.poly(np.ones(shift))
    reduced = np.r_[_binomial_norm(N - shift), np.zeros(shift)]
    for second in range(mmax + 1):
        indices = np.flatnonzero(orders[:, 1] == second)
        imax = min(nmax + order + 1, indices.size)
        source_count = min(indices.size, imax + shift)
        source = np.c_[work[mask][:, indices[:source_count]], np.zeros((mask.sum(), imax + shift - source_count))]
        if imax:
            out[np.ix_(mask, indices[:imax])] = convolve2d(source, kernel[None, :], mode="valid")[:, :imax] * reduced[:imax]
        if order:
            for first in range(nmax + 1):
                current = np.flatnonzero((orders[:, 0] == first) & (orders[:, 1] == second))
                following = np.flatnonzero((orders[:, 0] == first + 1) & (orders[:, 1] == second))
                if current.size and following.size:
                    out[mask, current[0]] += (shift / 4) * np.sqrt(first + 3) * out[mask, following[0]]
        out[np.ix_(mask, indices[min(nmax + 1, indices.size) :])] = 0
    if theta is None:
        out[np.ix_(mask, np.flatnonzero(orders[:, 1] > mmax))] = 0
    result = out.reshape(rotated.shape)
    if theta is not None:
        result = rdht(result, N, D, "inv", theta)
    return result, mask.reshape(arr.shape[:2])


__all__ = ["xdht2"]
