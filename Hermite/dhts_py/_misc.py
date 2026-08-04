"""Miscellaneous analysis and conversion utilities from the DHT toolbox."""

from __future__ import annotations

from math import comb, floor, log2
from typing import Sequence

import numpy as np
from scipy.signal import convolve2d

from ._core import dht2, dhtord, idht2


Array = np.ndarray


def bincoef(n, r) -> Array:
    """Binomial coefficients with the shape rules of ``bincoef.m``."""

    nn, rr = np.asarray(n, dtype=int), np.asarray(r, dtype=int)
    if np.any(nn < 0) or np.any(rr < 0):
        raise ValueError("n and r must be non-negative")
    if nn.size == rr.size:
        values = np.asarray([comb(int(a), int(b)) if b <= a else 0 for a, b in zip(nn.ravel(), rr.ravel())])
        return values.reshape(nn.shape)
    return np.asarray([[comb(int(a), int(b)) if b <= a else 0 for b in rr.ravel()] for a in nn.ravel()])


def samplat(p, N, T=None, shape: str = "full"):
    """Sampling lattice for single-scale or multiscale DHT coefficients."""

    positions = np.asarray(p)
    multiscale = T is None or isinstance(T, str)
    if isinstance(T, str):
        shape = T
    if multiscale:
        values = np.atleast_1d(N).astype(int)
        if values.size > 1:
            base_n, levels = int(values[0]), int(values[1])
        else:
            base_n, levels = 8, int(values[0])
        result = [samplat(positions, base_n, int(round(np.sqrt(base_n / 2))), shape)]
        for _ in range(1, levels):
            result.append(samplat(result[-1], 6, 2, shape))
        return result
    N, T = int(N), int(T)
    if positions.size < 2:
        step = 1
    else:
        step = positions[1] - positions[0]
    key = shape.lower()
    if key == "full":
        left_offsets = N / 2 - np.arange(N // 2)
        left = positions[0] - step * left_offsets
        right = positions[-1] + step * np.arange(1, N // 2 + 1)
        positions = np.r_[left, positions, right]
    elif key == "valid":
        positions = positions[N // 2 : positions.size - N // 2]
    elif key not in {"same", "symm", "repeat", "asymm", "cyclic"}:
        raise ValueError(f"invalid shape {shape!r}")
    return positions[::T]


def bt2dht(Y: Array, N: int, D: int | None = None) -> Array:
    """Map a 2-D Binomial Transform layout to DHT coefficient channels."""

    arr = np.asarray(Y, dtype=float).copy()
    T = int(N) + 1
    degree = N * N if D is None else int(D)
    if arr.shape[0] % T or arr.shape[1] % T:
        raise ValueError("both dimensions of Y must be multiples of N+1")
    weights = np.poly(np.ones(N))
    weights = np.sign(weights) * np.sqrt(np.abs(weights) / 2.0**N)
    arr *= np.tile(np.outer(weights, weights), (arr.shape[0] // T, arr.shape[1] // T))
    return np.stack(
        [arr[vertical::T, horizontal::T] for horizontal, vertical in dhtord(N, degree, 2)],
        axis=-1,
    )


def dht2bt(Z: Array, siz: Sequence[int], N: int, D: int | None = None) -> Array:
    """Map DHT coefficient channels to the 2-D Binomial Transform layout."""

    target = tuple(int(v) for v in siz[:2])
    T = int(N) + 1
    degree = N * N if D is None else int(D)
    if target[0] % T or target[1] % T:
        raise ValueError("siz dimensions must be multiples of N+1")
    out = np.zeros(target, dtype=float)
    for channel, (horizontal, vertical) in enumerate(dhtord(N, degree, 2)):
        out[vertical::T, horizontal::T] = np.asarray(Z)[..., channel]
    weights = np.poly(np.ones(N))
    weights = np.sign(weights) * np.sqrt(np.abs(weights) / 2.0**N)
    return out / np.tile(np.outer(weights, weights), (target[0] // T, target[1] // T))


def dob2(X: Array, Nc: int, Ns: int | None = None) -> Array:
    """Two-dimensional Difference of Binomial filtering."""

    central = 2 * int(Nc)
    surround = (4 * central if central > 0 else 4) if Ns is None else int(Ns)
    bc = np.poly(-np.ones(central)) / 2.0**central
    bs = np.poly(-np.ones(surround)) / 2.0**surround
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 2:
        return convolve2d(arr, np.outer(bc, bc), mode="same") - convolve2d(arr, np.outer(bs, bs), mode="same")
    return np.stack([dob2(arr[..., k], Nc, surround) for k in range(arr.shape[2])], axis=-1)


def edht(Y, k=3, *, return_bounds: bool = False):
    """Linear coefficient equalization/clipping from ``edht.m``."""

    if isinstance(Y, (list, tuple)):
        result = [edht(value, k) for value in Y]
        return (result, None, None) if return_bounds else result
    arr = np.asarray(Y, dtype=float)
    if arr.ndim >= 3 and arr.shape[2] > 1:
        result = np.stack([edht(arr[..., j], k) for j in range(arr.shape[2])], axis=2)
        return (result, None, None) if return_bounds else result
    bounds = np.atleast_1d(k).astype(float)
    if bounds.size > 1:
        ymin, ymax = float(bounds[0]), float(bounds[1])
    else:
        ymin, ymax = float(np.mean(arr) - bounds[0] * np.std(arr)), float(np.mean(arr) + bounds[0] * np.std(arr))
    result = arr.copy() if ymin == ymax else np.clip((arr - ymin) / (ymax - ymin), 0, 1)
    return (result, ymin, ymax) if return_bounds else result


def _histeq(image: Array, bins: int = 256) -> Array:
    arr = np.asarray(image, dtype=float)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if lo == hi:
        return np.zeros_like(arr)
    normalized = np.clip((arr - lo) / (hi - lo), 0, 1)
    hist, edges = np.histogram(normalized.ravel(), bins=bins, range=(0, 1))
    cdf = hist.cumsum().astype(float)
    nonzero = np.flatnonzero(cdf)
    if not nonzero.size or cdf[-1] == cdf[nonzero[0]]:
        return normalized
    cdf = (cdf - cdf[nonzero[0]]) / (cdf[-1] - cdf[nonzero[0]])
    return np.interp(normalized.ravel(), edges[:-1], cdf).reshape(arr.shape)


def equaliz(I: Array) -> Array:
    """DHT-based equalization from ``equaliz.m``."""

    from ._multiscale import imdht2, mdht2

    image = np.asarray(I, dtype=float)
    levels = max(1, floor(log2(min(image.shape[:2])) / 2))
    coefficients = mdht2(_histeq(image), 4, levels, "symm")
    coefficients[-1][..., 0] = 0
    reconstructed = imdht2(coefficients, image.shape, 4, "symm")
    lo, hi = np.nanmin(reconstructed), np.nanmax(reconstructed)
    return np.zeros_like(reconstructed) if lo == hi else (reconstructed - lo) / (hi - lo)


def imcorn(X: Array, N: int = 2, kappa: float = 0.1) -> Array:
    """Harris-like corner response based on DHT derivatives."""

    # The MATLAB source unconditionally sets N=2; keep that historical rule.
    Y = dht2(np.asarray(X, dtype=float), 2, 1, 1, "repeat")
    products = np.stack((Y[..., 1] * Y[..., 2], Y[..., 1] ** 2, Y[..., 2] ** 2), axis=-1)
    smooth = np.stack([dht2(products[..., k], 2, 0, 1, "repeat")[..., 0] for k in range(3)], axis=-1)
    return smooth[..., 1] * smooth[..., 2] - smooth[..., 0] ** 2 - kappa * (smooth[..., 1] + smooth[..., 2]) ** 2


def dhtentr(Y, L: int = 255):
    """Histogram entropy of every coefficient channel."""

    if isinstance(Y, (list, tuple)):
        return [dhtentr(value, L) for value in Y]
    arr = np.asarray(Y, dtype=float)
    if arr.ndim == 2:
        arr = arr[..., None]
    values = []
    for channel in range(arr.shape[2]):
        shifted = arr[..., channel] + (0.5 if channel > 0 else 0.0)
        counts, _ = np.histogram(np.clip(shifted, 0, 1), bins=int(L), range=(0, 1))
        total = counts.sum()
        probability = counts[counts > 0] / total
        values.append(float(-np.sum(probability * np.log2(probability))))
    return np.asarray(values)


def _neighbors8(X: Array, connectivity: int) -> Array:
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    padded = np.pad(X, 1, mode="edge")
    return np.stack([padded[1 + dr : 1 + dr + X.shape[0], 1 + dc : 1 + dc + X.shape[1]] for dr, dc in offsets], axis=-1)


def zcross(Y: Array, N: int, D: int, conn: int = 8, CMIN=None, SMAX=None, *, scale_model: Array | None = None):
    """Zero-crossing mask from rotated third-order coefficients.

    The optional scale/contrast branch of the MATLAB function requires the
    absent file ``scest.mat``.  Supply its 2-by-scale coefficient matrix as
    ``scale_model`` to reproduce that branch.
    """

    if min(D, N) < 3:
        raise ValueError("zcross requires coefficients through third order")
    from ._steering import rdht

    rotated = rdht(np.asarray(Y)[..., :10], N, 3, "fwd", "grad")
    center = rotated[..., 3]
    neighbors = _neighbors8(center, int(conn))
    crossing = np.any((neighbors * center[..., None] < 0) & (np.abs(center[..., None]) <= np.abs(neighbors)), axis=-1)
    mask = crossing & (rotated[..., 6] < 0)
    if CMIN is None and SMAX is None:
        return mask
    if scale_model is None:
        raise FileNotFoundError("scale/contrast estimation needs scest.mat; pass its matrix with scale_model=")
    model = np.asarray(scale_model, dtype=float)
    column = N // 2 - 2
    scale = N * (model[0, column] * (rotated[..., 1] / rotated[..., 6]) + model[1, column])
    scale *= scale > 0
    contrast = 2.5 * rotated[..., 1] * np.sqrt(1 + scale * (8 / N))
    if SMAX is not None:
        mask &= (scale > 0) & (scale <= SMAX)
    if CMIN is not None:
        mask &= contrast >= CMIN
    return mask, scale, contrast


def hermiteFiltersFreq(X, N: int, D: int, Tsub: int, rotate=None) -> Array:
    """Generate the DHT response to an impulse or supplied image."""

    if np.isscalar(X) and float(X) == 0:
        image = np.zeros((N, N), dtype=float)
        image[0, 0] = 1
    else:
        image = np.asarray(X, dtype=float)
    return dht2(image, N, D, Tsub, " " if rotate is None else rotate)


def obtainOrdCoefs(Y: Array, N: int, D: int) -> dict[str, Array]:
    """Reorder coefficient maps in row-major order of their 2-D order table."""

    arr = np.asarray(Y)
    table = np.zeros((min(N, D) + 1, min(N, D) + 1), dtype=int)
    for channel, (horizontal, vertical) in enumerate(dhtord(N, D, 2), start=1):
        table[vertical, horizontal] = channel
    indices = table[table > 0] - 1
    return {"coefs": arr[..., indices], "mat": table}
