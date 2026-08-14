"""Private helpers shared by direct MATLAB-to-Python translations.

Public toolbox routines live in their MATLAB-homologous modules.  This file
contains only Python-specific plumbing that would otherwise be duplicated
across several translations: argument validation, padding, convolution-axis
handling, inverse-transform geometry, and the recursive FBT kernel.
"""

from __future__ import annotations

from math import ceil, floor
from typing import Sequence

import numpy as np
from scipy.signal import convolve2d


Array = np.ndarray
BOUNDARY_MODES = {"full", "same", "valid", "symm", "repeat", "asymm", "cyclic"}


def integer(name: str, value: int, minimum: int = 0) -> int:
    if not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def shape_tuple(shape: int | Sequence[int]) -> tuple[int, ...]:
    if np.isscalar(shape):
        return (int(shape),)
    return tuple(int(v) for v in shape)


def scale_degree(value, default_n: int) -> tuple[int, int]:
    """Parse MATLAB's scalar-or-[N,D] multiscale convention."""

    values = np.atleast_1d(value).astype(int)
    return (int(values[0]), int(values[1])) if values.size > 1 else (default_n, int(values[0]))


def convmtx(vector: Array, columns: int) -> Array:
    vector = np.asarray(vector, dtype=float).ravel()
    out = np.zeros((vector.size + columns - 1, columns), dtype=float)
    for j in range(columns):
        out[j : j + vector.size, j] = vector
    return out


def pad_axis(X: Array, N: int, axis: int, mode: str) -> Array:
    left = ceil(N / 2)
    right = floor(N / 2)
    if N == 0:
        return X
    pads = [(0, 0)] * X.ndim
    pads[axis] = (left, right)
    if mode == "repeat":
        return np.pad(X, pads, mode="edge")
    if mode == "cyclic":
        return np.pad(X, pads, mode="wrap")
    if mode == "symm":
        # dht.m excludes the edge sample itself, unlike dht2.m.
        return np.pad(X, pads, mode="reflect")
    if mode == "asymm":
        moved = np.moveaxis(X, axis, 0)
        left_indices = np.arange(left - 1, -1, -1)
        right_indices = moved.shape[0] - 1 - np.arange(right)
        extended = np.concatenate(
            (
                2 * moved[:1] - moved[left_indices],
                moved,
                2 * moved[-1:] - moved[right_indices],
            ),
            axis=0,
        )
        return np.moveaxis(extended, 0, axis)
    raise ValueError(f"unsupported boundary mode {mode!r}")


def pad_spatial(X: Array, N: int, mode: str, dimensions: int) -> Array:
    width = ceil(N / 2)
    pads = [(width, width) for _ in range(dimensions)] + [(0, 0)] * (X.ndim - dimensions)
    if mode == "symm":
        return np.pad(X, pads, mode="symmetric")
    if mode == "repeat":
        return np.pad(X, pads, mode="edge")
    if mode == "asymm":
        if dimensions != 2 or X.ndim != 2:
            result = X
            for axis in range(dimensions):
                result = pad_axis(result, N, axis, "asymm")
            return result
        rows, cols = X.shape
        bc = np.arange(width - 1, -1, -1)
        rr = rows - 1 - np.arange(width)
        cc = cols - 1 - np.arange(width)
        top = np.concatenate(
            (
                2 * X[0, 0] - X[np.ix_(bc, bc)],
                2 * X[:1, :] - X[bc, :],
                2 * X[0, -1] - X[np.ix_(bc, cc)],
            ),
            axis=1,
        )
        middle = np.concatenate(
            (
                2 * X[:, :1] - X[:, bc],
                X,
                2 * X[:, -1:] - X[:, cc],
            ),
            axis=1,
        )
        bottom = np.concatenate(
            (
                2 * X[-1, 0] - X[np.ix_(rr, bc)],
                2 * X[-1:, :] - X[rr, :],
                2 * X[-1, -1] - X[np.ix_(rr, cc)],
            ),
            axis=1,
        )
        return np.concatenate((top, middle, bottom), axis=0)
    if mode == "cyclic":
        return np.pad(X, pads, mode="wrap")
    raise ValueError(f"unsupported boundary mode {mode!r}")


def conv_axis(X: Array, kernel: Array, axis: int, mode: str) -> Array:
    moved = np.moveaxis(np.asarray(X, dtype=float), axis, 0)
    original_tail = moved.shape[1:]
    flat = moved.reshape(moved.shape[0], -1)
    result = convolve2d(flat, np.asarray(kernel, dtype=float)[:, None], mode=mode)
    result = result.reshape((result.shape[0],) + original_tail)
    return np.moveaxis(result, 0, axis)


def default_dim(shape: Sequence[int]) -> int:
    for index, size in enumerate(shape):
        if size > 1:
            return index + 1
    raise ValueError("input must have a non-singleton dimension")


def synthesis_filters(N: int, D: int, T: int, low_resolution: bool = False) -> Array:
    from .dhtmtx import dhtmtx

    if T <= 2:
        G = T * dhtmtx(N, D)[::-1, :]
    else:
        _, G = dhtmtx(N, D, T)
    if low_resolution:
        G = G * np.sqrt(0.75 ** np.arange(G.shape[1]))[None, :]
    return G


def synthesis_geometry(target: int, N: int, shape: str) -> tuple[int, str]:
    if shape == "full":
        return target + N, "valid"
    if shape == "same":
        return target, "same"
    if shape == "valid":
        length = target - N
        if length <= 0:
            raise ValueError("target is too small for shape='valid'")
        return length, "full"
    if shape in {"symm", "repeat", "asymm", "cyclic"}:
        return target + N, "valid"
    raise ValueError(f"invalid shape {shape!r}")


def expand_1d_coefficients(Y: Array, target: int, N: int, T: int, shape: str) -> tuple[Array, int]:
    if shape not in {"symm", "repeat", "asymm", "cyclic"}:
        return Y, 0
    ylen = target + N
    left = floor(ceil(N / 2) / T)
    t0 = ceil(N / 2) - T * left
    zlen = len(range(t0, ylen, T))
    right = zlen - Y.shape[0] - left
    if right < 0:
        raise ValueError("coefficient dimensions are incompatible with boundary mode")
    out = np.zeros((zlen,) + Y.shape[1:], dtype=float)
    if shape in {"repeat", "asymm"}:
        indices = np.r_[np.zeros(left, int), np.arange(Y.shape[0]), np.full(right, Y.shape[0] - 1)]
    elif shape == "cyclic":
        indices = np.arange(-left, Y.shape[0] + right) % Y.shape[0]
    else:
        indices = np.r_[
            np.arange(left - 1, -1, -1),
            np.arange(Y.shape[0]),
            np.arange(Y.shape[0] - 1, Y.shape[0] - right - 1, -1),
        ]
    out[..., 0] = Y[indices, ..., 0]
    if Y.shape[-1] > 1:
        out[left : zlen - right if right else zlen, ..., 1:] = Y[..., 1:]
    return out, t0


def parse_dht2_options(cod: str | None, args: tuple) -> tuple[str, str, tuple]:
    shape = "full"
    code = cod or " "
    if len(code) > 3:
        shape = code.lower()
        code = str(args[0]) if args else " "
        args = args[1:] if args else ()
    elif args and isinstance(args[0], str) and args[0].lower() in BOUNDARY_MODES:
        shape = args[0].lower()
        args = args[1:]
    return shape, code or " ", args


def expand_coefficients_2d(
    Y: Array, xsiz: tuple[int, int], N: int, T: int, shape: str
) -> tuple[Array, tuple[int, int], int, str]:
    if shape == "full":
        return Y, (xsiz[0] + N, xsiz[1] + N), 0, "valid"
    if shape == "same":
        return Y, xsiz, 0, "same"
    if shape == "valid":
        ysiz = (xsiz[0] - N, xsiz[1] - N)
        if min(ysiz) <= 0:
            raise ValueError("xsiz is too small for shape='valid'")
        return Y, ysiz, 0, "full"
    if shape not in {"symm", "repeat", "asymm", "cyclic"}:
        raise ValueError(f"invalid shape {shape!r}")
    ysiz = (xsiz[0] + N, xsiz[1] + N)
    left = floor(ceil(N / 2) / T)
    t0 = ceil(N / 2) - T * left
    zsiz = (len(range(t0, ysiz[0], T)), len(range(t0, ysiz[1], T)))
    right = (zsiz[0] - Y.shape[0] - left, zsiz[1] - Y.shape[1] - left)
    if min(right) < 0:
        raise ValueError("coefficient dimensions are incompatible with boundary mode")

    def indices(length: int, after: int) -> Array:
        if shape in {"repeat", "asymm"}:
            return np.r_[np.zeros(left, int), np.arange(length), np.full(after, length - 1)]
        if shape == "cyclic":
            return np.arange(-left, length + after) % length
        return np.r_[
            np.arange(left - 1, -1, -1),
            np.arange(length),
            np.arange(length - 1, length - after - 1, -1),
        ]

    rows, cols = indices(Y.shape[0], right[0]), indices(Y.shape[1], right[1])
    out = np.zeros(zsiz + Y.shape[2:], dtype=float)
    out[..., 0] = Y[..., 0][np.ix_(rows, cols)]
    if Y.shape[2] > 1:
        rend = zsiz[0] - right[0] if right[0] else zsiz[0]
        cend = zsiz[1] - right[1] if right[1] else zsiz[1]
        out[left:rend, left:cend, 1:] = Y[..., 1:]
    return out, ysiz, t0, "valid"


def rbt_vector(vector: Array, fnorm: float) -> Array:
    vector = np.asarray(vector, dtype=float)
    T = vector.shape[0]
    if T <= 1:
        return vector.copy()
    half = T // 2
    y0, y1 = vector.copy(), vector.copy()
    for _ in range(half):
        y0 = y0[:-1] + y0[1:]
        y1 = y1[:-1] - y1[1:]
    y1 = y1 / fnorm**half
    if T % 2:
        y0 = (y0[:-1] + y0[1:]) / (fnorm ** (half + 1))
    else:
        y0 = y0 / fnorm**half
    return np.concatenate((rbt_vector(y0, fnorm), rbt_vector(y1, fnorm)), axis=0)
