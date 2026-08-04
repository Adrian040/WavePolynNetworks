"""Numerical core of the Discrete Hermite Transform toolbox.

The routines in this module are close translations of the 2005 MATLAB DHT
toolbox by J. L. Silvan-Cardenas.  MATLAB dimensions passed as ``dim`` remain
one-based; coefficient channels are stored on the last NumPy axis.
"""

from __future__ import annotations

from functools import lru_cache
from math import ceil, comb, factorial, floor
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import convolve as signal_convolve
from scipy.signal import convolve2d


Array = np.ndarray
_BOUNDARY_MODES = {"full", "same", "valid", "symm", "repeat", "asymm", "cyclic"}


def _integer(name: str, value: int, minimum: int = 0) -> int:
    if not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _shape_tuple(shape: int | Sequence[int]) -> tuple[int, ...]:
    if np.isscalar(shape):
        return (int(shape),)
    return tuple(int(v) for v in shape)


def _convmtx(vector: Array, columns: int) -> Array:
    vector = np.asarray(vector, dtype=float).ravel()
    out = np.zeros((vector.size + columns - 1, columns), dtype=float)
    for j in range(columns):
        out[j : j + vector.size, j] = vector
    return out


def hermite(D: int | Iterable[int], x: Array | Sequence[float] | None = None) -> Array:
    """Return physicists' Hermite polynomials, matching ``hermite.m``.

    With ``x=None`` the columns contain polynomial coefficients in descending
    powers, bottom-aligned as in MATLAB.  Otherwise the columns contain values
    evaluated at every element of ``x``.
    """

    degrees = np.atleast_1d(D).astype(int)
    if degrees.size == 0 or np.any(degrees < 0):
        raise ValueError("D must contain non-negative polynomial degrees")
    dmax = int(degrees.max())
    if x is None:
        result = np.zeros((dmax + 1, degrees.size), dtype=float)
    else:
        xx = np.asarray(x, dtype=float).ravel()
        result = np.zeros((xx.size, degrees.size), dtype=float)

    for column, n in enumerate(degrees):
        coeff = np.zeros(n + 1, dtype=float)
        for j in range(n // 2 + 1):
            power = n - 2 * j
            coeff[n - power] = (
                (-1.0) ** j
                * factorial(n)
                / (factorial(j) * factorial(power))
                * 2.0**power
            )
        if x is None:
            result[-(n + 1) :, column] = coeff
        else:
            result[:, column] = np.polyval(coeff, xx)
    return result


def dhtmtx(N: int, D: int | None = None, T: int | None = None):
    """Construct DHT analysis filters and optional interpolation filters.

    ``dhtmtx(N, D)`` returns ``H``.  Supplying ``T`` returns ``(H, G)``, just
    as requesting two outputs from MATLAB ``dhtmtx``.
    """

    N = _integer("N", N)
    D = N if D is None else min(N, _integer("D", D))
    if N == 0:
        H = np.ones((1, 1), dtype=float)
        return (H, H.copy()) if T is not None else H

    smooth = np.array([0.5, 0.5], dtype=float)
    detail = np.array([0.5, -0.5], dtype=float)
    H = np.column_stack((smooth, detail)) if D > 0 else smooth[:, None]
    for _ in range(2, N + 1):
        old_columns = H.shape[1]
        smoothed = np.column_stack(
            [np.convolve(smooth, H[:, j], mode="full") for j in range(old_columns)]
        )
        if old_columns <= D:
            H = np.column_stack((smoothed, np.convolve(detail, H[:, -1], mode="full")))
        else:
            H = smoothed

    constants = 2.0 ** (N / 2.0) * np.sqrt(H[: D + 1, 0])
    H *= constants[None, :]
    if T is None:
        return H

    T = _integer("T", T, 1)
    if T > N:
        raise ValueError("T must satisfy 1 <= T <= N")
    W = _convmtx(H[:, 0], N + 1)
    start = N - floor(N / T) * T
    weights = W[np.arange(start, 2 * N + 1, T), :].sum(axis=0)
    if np.any(weights == 0):
        raise ZeroDivisionError("zero interpolation weight in dhtmtx")
    G = H[::-1, :] / weights[:, None]
    return H, G


def gbtmtx(N: int, theta: float, D: int | None = None, T: int | None = None):
    """Generalized binomial transform matrix (the intended ``gbtmtx.m`` API)."""

    N = _integer("N", N)
    D = N if D is None else min(N, _integer("D", D))
    if N == 0:
        H = np.ones((1, 1), dtype=float)
        return (H, H.copy()) if T is not None else H
    smooth = np.array([np.sin(theta), np.cos(theta)], dtype=float)
    detail = np.array([np.cos(theta), -np.sin(theta)], dtype=float)
    H = np.column_stack((smooth, detail)) if D > 0 else smooth[:, None]
    for _ in range(2, N + 1):
        ncol = H.shape[1]
        smoothed = np.column_stack([np.convolve(smooth, H[:, j]) for j in range(ncol)])
        H = (
            np.column_stack((smoothed, np.convolve(detail, H[:, -1])))
            if ncol <= D
            else smoothed
        )
    if T is None:
        return H
    T = _integer("T", T, 1)
    if T > N:
        raise ValueError("T must satisfy 1 <= T <= N")
    W = _convmtx(H[:, 0], N + 1)
    start = N - floor(N / T) * T
    weights = W[np.arange(start, 2 * N + 1, T), :].sum(axis=0)
    G = H[::-1, :] / weights[:, None]
    return H, G


def chtmtx(s: float, D: int, x: Array | Sequence[float] | None = None) -> Array:
    """Continuous Hermite transform functions from ``chtmtx.m``."""

    if s <= 0:
        raise ValueError("s must be positive")
    D = _integer("D", D)
    xx = (-4 * s + np.arange(floor(8 * s) + 1)) if x is None else np.asarray(x, dtype=float).ravel()
    n = np.arange(D + 1)
    c = (-1.0) ** n * np.sqrt(2.0**n * np.array([factorial(int(v)) for v in n]))
    gaussian = np.exp(-(xx**2) / (4 * s)) / np.sqrt(4 * s * np.pi)
    polynomials = hermite(n, xx / np.sqrt(4 * s))
    return polynomials * gaussian[:, None] / c[None, :]


def dhtord(N: int, D: int, K: int = 2) -> Array:
    """Return coefficient orders in the exact channel order of the toolbox."""

    N = _integer("N", N)
    D = _integer("D", D)
    if K == 1:
        return np.arange(min(D, N) + 1, dtype=int)[:, None]
    if K == 2:
        return np.asarray(
            [
                (total - m, m)
                for total in range(min(D, 2 * N) + 1)
                for m in range(max(0, total - N), min(N, total) + 1)
            ],
            dtype=int,
        )
    if K == 3:
        return np.asarray(
            [
                (b, a, c)
                for c in range(min(D, N) + 1)
                for total in range(min(D - c, 2 * N) + 1)
                for a in range(max(0, total - N), min(N, total) + 1)
                for b in [total - a]
            ],
            dtype=int,
        )
    raise ValueError("K must be 1, 2, or 3")


def _pad_axis(X: Array, N: int, axis: int, mode: str) -> Array:
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
        # dht.m excludes the edge itself, unlike dht2.m.
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


def _pad_spatial(X: Array, N: int, mode: str, dimensions: int) -> Array:
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
                result = _pad_axis(result, N, axis, "asymm")
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


def _conv_axis(X: Array, kernel: Array, axis: int, mode: str) -> Array:
    moved = np.moveaxis(np.asarray(X, dtype=float), axis, 0)
    original_tail = moved.shape[1:]
    flat = moved.reshape(moved.shape[0], -1)
    result = convolve2d(flat, np.asarray(kernel, dtype=float)[:, None], mode=mode)
    result = result.reshape((result.shape[0],) + original_tail)
    return np.moveaxis(result, 0, axis)


def _default_dim(shape: Sequence[int]) -> int:
    for index, size in enumerate(shape):
        if size > 1:
            return index + 1
    raise ValueError("input must have a non-singleton dimension")


def dht(
    X: Array,
    N: int,
    D: int,
    T: int,
    dim: int | None = None,
    shape: str = "full",
    cod: str = "h",
    *args,
    return_lowpass: bool = False,
):
    """One-dimensional DHT along MATLAB-style dimension ``dim``.

    Set ``return_lowpass=True`` to obtain ``(highpass, lowpass)`` as with two
    MATLAB output arguments.
    """

    X = np.asarray(X, dtype=float)
    N = _integer("N", N)
    D = min(_integer("D", D), N)
    T = _integer("T", T, 1)
    if N and T > N:
        raise ValueError("T must satisfy 1 <= T <= N")
    if dim is None:
        dim = _default_dim(X.shape)
    axis = int(dim) - 1
    if not 0 <= axis < X.ndim:
        raise ValueError("dim is outside the input dimensions")
    shape = shape.lower()
    if shape not in _BOUNDARY_MODES:
        raise ValueError(f"invalid shape {shape!r}")

    work = X
    conv_mode = shape
    if shape in {"symm", "repeat", "asymm", "cyclic"}:
        work = _pad_axis(work, N, axis, shape)
        conv_mode = "valid"
    H = dhtmtx(N, D)
    code = cod or " "
    if code[0].lower() == "l":
        H = H / np.sqrt(0.75 ** np.arange(H.shape[1]))[None, :]
        code = code[1:] or " "
    elif code[0].lower() == "h":
        code = code[1:] or " "

    channels = []
    for order in range(D + 1):
        filtered = _conv_axis(work, H[:, order], axis, conv_mode)
        channels.append(filtered.take(range(0, filtered.shape[axis], T), axis=axis))
    Y = np.stack(channels, axis=-1)
    if code and code[0].lower() == "s":
        if not args:
            raise ValueError("cod='s' requires tau")
        tau = np.asarray(args[0], dtype=float)
        for order in range(1, D + 1):
            Y[..., order] *= tau**order

    if return_lowpass:
        low = Y[..., 0]
        high = Y if "0" in code else Y[..., 1:]
        return high, low
    return Y


def _synthesis_filters(N: int, D: int, T: int, low_resolution: bool = False) -> Array:
    if T <= 2:
        G = T * dhtmtx(N, D)[::-1, :]
    else:
        _, G = dhtmtx(N, D, T)
    if low_resolution:
        G = G * np.sqrt(0.75 ** np.arange(G.shape[1]))[None, :]
    return G


def _synthesis_geometry(target: int, N: int, shape: str) -> tuple[int, str]:
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


def _expand_1d_coefficients(Y: Array, target: int, N: int, T: int, shape: str) -> tuple[Array, int]:
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
    if shape == "repeat" or shape == "asymm":
        indices = np.r_[np.zeros(left, int), np.arange(Y.shape[0]), np.full(right, Y.shape[0] - 1)]
    elif shape == "cyclic":
        indices = np.arange(-left, Y.shape[0] + right) % Y.shape[0]
    else:
        indices = np.r_[np.arange(left - 1, -1, -1), np.arange(Y.shape[0]), np.arange(Y.shape[0] - 1, Y.shape[0] - right - 1, -1)]
    out[..., 0] = Y[indices, ..., 0]
    if Y.shape[-1] > 1:
        out[left : zlen - right if right else zlen, ..., 1:] = Y[..., 1:]
    return out, t0


def idht(
    Y: Array,
    xsiz: int | Sequence[int],
    N: int,
    D: int,
    T: int,
    dim: int | None = None,
    cod: str = " ",
    *args,
    shape: str | None = None,
) -> Array:
    """Inverse one-dimensional DHT."""

    target_shape = _shape_tuple(xsiz)
    if dim is None:
        dim = _default_dim(target_shape)
    axis = int(dim) - 1
    if not 0 <= axis < len(target_shape):
        raise ValueError("dim is outside xsiz")
    # MATLAB overloads COD with a long shape string.
    if shape is None and isinstance(cod, str) and len(cod) > 3:
        shape, cod = cod.lower(), (str(args[0]) if args else " ")
        args = args[1:] if args else ()
    shape = (shape or "full").lower()
    code = cod or " "
    low_resolution = code[0].lower() == "l"
    if code[0].lower() in {"l", "h"}:
        code = code[1:] or " "
    D = min(_integer("D", D), _integer("N", N))
    T = _integer("T", T, 1)
    G = _synthesis_filters(N, D, T, low_resolution)

    arr = np.asarray(Y, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    # For a pure 1-D signal, sample and coefficient axes are already 0,-1.
    if len(target_shape) > 1:
        arr = np.moveaxis(arr, axis, 0)
    if arr.shape[-1] < D + 1:
        raise ValueError(f"expected at least {D + 1} coefficient channels")
    if code[0].lower() == "s":
        if not args:
            raise ValueError("cod='s' requires tau")
        tau = np.asarray(args[0], dtype=float)
        arr = arr.copy()
        for order in range(1, D + 1):
            arr[..., order] *= tau**order

    arr, t0 = _expand_1d_coefficients(arr, target_shape[axis], N, T, shape)
    ylen, conv_mode = _synthesis_geometry(target_shape[axis], N, shape)
    flat_tail = arr.shape[1:-1]
    flat = arr.reshape(arr.shape[0], -1, arr.shape[-1])
    result = np.zeros((target_shape[axis], flat.shape[1]), dtype=float)
    lattice = np.arange(t0, ylen, T)
    if len(lattice) != flat.shape[0]:
        raise ValueError("coefficient dimensions do not match N, T, shape, and xsiz")
    for order in range(D + 1):
        up = np.zeros((ylen, flat.shape[1]), dtype=float)
        up[lattice, :] = flat[:, :, order]
        result += convolve2d(up, G[:, order, None], mode=conv_mode)
    moved_shape = (target_shape[axis],) + tuple(target_shape[i] for i in range(len(target_shape)) if i != axis)
    result = result.reshape(moved_shape)
    return np.moveaxis(result, 0, axis) if len(target_shape) > 1 else result.ravel()


def dhti(
    Y: Array,
    xsiz: int | Sequence[int],
    N: int,
    D: int,
    T: int,
    dim: int | None = None,
    sumopt: str = "nosum",
    cod: str = " ",
    *args,
    shape: str | None = None,
) -> Array:
    """Interpolate 1-D coefficient contributions without necessarily summing."""

    arr = np.asarray(Y, dtype=float)
    ncoeff = arr.shape[-1] if arr.ndim > 1 else 1
    D1 = min(D, N)
    extra = max(0, ncoeff - (D1 + 1))
    pieces: list[Array] = []
    for channel in range(ncoeff):
        order = 0 if channel <= extra else channel - extra
        one = np.zeros(arr.shape[:-1] + (D1 + 1,), dtype=float)
        one[..., order] = arr[..., channel]
        pieces.append(idht(one, xsiz, N, D1, T, dim, cod, *args, shape=shape))
    if sumopt.lower() == "nosum":
        return np.stack(pieces, axis=-1)
    if sumopt.lower() == "high":
        low_count = extra + 1
        low = np.sum(pieces[:low_count], axis=0)
        high = np.sum(pieces[low_count:], axis=0) if low_count < len(pieces) else np.zeros_like(low)
        return np.stack((low, high), axis=-1)
    if sumopt.lower() == "predict":
        raise NotImplementedError("SUMOPT='predict' was not implemented in dhti.m")
    raise ValueError(f"unrecognized sumopt {sumopt!r}")


def _parse_dht2_options(cod: str | None, args: tuple) -> tuple[str, str, tuple]:
    shape = "full"
    code = cod or " "
    if len(code) > 3:
        shape = code.lower()
        code = str(args[0]) if args else " "
        args = args[1:] if args else ()
    elif args and isinstance(args[0], str) and args[0].lower() in _BOUNDARY_MODES:
        shape = args[0].lower()
        args = args[1:]
    return shape, code or " ", args


def dht2(X: Array, N: int, D: int, T: int, cod: str = " ", *args, shape: str | None = None, return_aux: bool = False):
    """Two-dimensional discrete Hermite transform.

    The output is ``(rows, columns, coefficients)`` or
    ``(rows, columns, coefficients, bands)`` for multiband input.
    """

    parsed_shape, code, args = _parse_dht2_options(cod, args)
    shape = (shape or parsed_shape).lower()
    X = np.asarray(X, dtype=float)
    if X.ndim == 3 and X.shape[2] > 1:
        bands = [dht2(X[..., k], N, D, T, shape, code, *args, return_aux=return_aux) for k in range(X.shape[2])]
        if return_aux:
            return np.stack([v[0] for v in bands], axis=-1), [v[1] for v in bands]
        return np.stack(bands, axis=-1)
    if X.ndim != 2:
        raise ValueError("dht2 expects a 2-D image or a 3-D multiband image")
    N = _integer("N", N)
    D = min(_integer("D", D), 2 * N)
    T = _integer("T", T, 1)
    H = dhtmtx(N, D)
    work = X
    conv_mode = shape
    if shape in {"symm", "repeat", "asymm", "cyclic"}:
        work = _pad_spatial(X, N, shape, 2)
        conv_mode = "valid"
    elif shape not in {"full", "same", "valid"}:
        raise ValueError(f"invalid shape {shape!r}")

    multiscale_flag = 0
    if code[0].lower() == "l":
        H = H / np.sqrt(0.75 ** np.arange(H.shape[1]))[None, :]
        multiscale_flag = 1 + int(code[0] == "L")
        code = code[1:] or " "
    elif code[0].lower() == "h":
        code = code[1:] or " "

    maps = []
    for horizontal, vertical in dhtord(N, D, 2):
        kernel = np.outer(H[:, vertical], H[:, horizontal])
        maps.append(convolve2d(work, kernel, mode=conv_mode)[::T, ::T])
    Y = np.stack(maps, axis=-1)
    if not code.strip():
        return (Y, None) if return_aux else Y

    from . import _steering as post

    key = code[0]
    aux = None
    if key == "q":
        Y = post.qdht(Y, "fwd", code[1:], N, D, *args)
    elif key == "r":
        Y, aux = post.rdht(Y, N, D, "fwd", *(args or ("grad",)), return_theta=True)
    elif key == "d":
        Y, aux = post.ddht(Y, N, D, "fwd", *(args or ("grad",)), return_theta=True)
    elif key == "s":
        Y, aux, _ = post.sdht2(Y, N, D, *args)
    elif key == "c":
        Y, aux, _ = post.sdht2(Y, N, D, None, None)
    elif key == "m":
        Y, aux = post.dhtmorph(Y, N, D, *args)
    elif key == "G":
        offset = ceil(N / T)
        inner = Y[offset:-offset, offset:-offset, 0] if offset else Y[..., 0]
        mean, std = float(np.mean(inner)), float(np.std(inner))
        Y[..., 0] = 0.5 if std == 0 else (1 + np.tanh((Y[..., 0] - mean) / std)) / 2
    elif key == "i":
        values, aux = post.dhtgi(Y, N, D, *(args or ()))
        Y = np.concatenate((Y[..., :1], values), axis=-1)
    elif key == "g":
        out = np.zeros(Y.shape[:2] + (D + 1,), dtype=float)
        out[..., 0] = Y[..., 0]
        for order in range(1, min(D, N) + 1):
            a, b = post.gauge(Y, N, D, order, components=True)
            out[..., order] = np.hypot(a, b)
        Y = out
    elif key == "e":
        from ._misc import edht

        Y[..., 0] = edht(Y[..., 0], *(args or ()))
    elif key == "x":
        Y, aux = post.xdht2(Y, N, D, *args)
    elif key == "E":
        if D < 1:
            raise ValueError("erosion requires D > 0")
        grad = np.hypot(Y[..., 1], Y[..., 2])
        mask = grad > args[0] if args else np.ones(Y.shape[:2], bool)
        if args and multiscale_flag < 2:
            Y, _, _ = post.sdht2(Y, N, D, (~mask).astype(float))
        Y[..., 0][mask] -= grad[mask] / np.sqrt(2.0)
        aux = mask
    return (Y, aux) if return_aux else Y


def _expand_coefficients_2d(Y: Array, xsiz: tuple[int, int], N: int, T: int, shape: str) -> tuple[Array, tuple[int, int], int, str]:
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
        return np.r_[np.arange(left - 1, -1, -1), np.arange(length), np.arange(length - 1, length - after - 1, -1)]

    rows, cols = indices(Y.shape[0], right[0]), indices(Y.shape[1], right[1])
    out = np.zeros(zsiz + Y.shape[2:], dtype=float)
    out[..., 0] = Y[..., 0][np.ix_(rows, cols)]
    if Y.shape[2] > 1:
        rend = zsiz[0] - right[0] if right[0] else zsiz[0]
        cend = zsiz[1] - right[1] if right[1] else zsiz[1]
        out[left:rend, left:cend, 1:] = Y[..., 1:]
    return out, ysiz, t0, "valid"


def idht2(Y: Array, xsiz: Sequence[int], N: int, D: int, T: int, cod: str = " ", *args, shape: str | None = None, return_aux: bool = False):
    """Inverse two-dimensional DHT."""

    parsed_shape, code, args = _parse_dht2_options(cod, args)
    shape = (shape or parsed_shape).lower()
    target = tuple(int(v) for v in xsiz[:2])
    arr = np.asarray(Y, dtype=float)
    if len(xsiz) == 3:
        bands = [idht2(arr[..., k], target, N, D, T, shape, code, *args, return_aux=return_aux) for k in range(int(xsiz[2]))]
        if return_aux:
            return np.stack([v[0] for v in bands], axis=-1), [v[1] for v in bands]
        return np.stack(bands, axis=-1)
    if arr.ndim == 2:
        arr = arr[..., None]
    quincunx = "q" in code.lower()
    low_resolution = code[0].lower() == "l"
    if code[0].lower() in {"l", "h"}:
        code = code[1:] or " "
    D = min(_integer("D", D), 2 * _integer("N", N))
    T = _integer("T", T, 1)
    if quincunx:
        T2 = 2 * T
        Hq, Gq = dhtmtx(N, min(D, N), T2)
        Hq = Hq[::-1, :]
        start = ceil((N + 1) / 2) - T * ceil(N / T2) + ceil(N / 2)
        stop = ceil((N + 1) / 2) + T2 - T * ceil(N / T2) + ceil(N / 2) - 1
        indices = np.mod(np.arange(start, stop + 1) - 1, N).astype(int)
        ratio = Hq[indices, 0] / Gq[indices, 0]
        base_weight = np.outer(ratio, ratio)
        base_weight = base_weight + np.fft.fftshift(base_weight)
        repetitions = (ceil(target[0] / T2), ceil(target[1] / T2))
        quincunx_weight = np.tile(base_weight, repetitions)[: target[0], : target[1]]
        G = Hq
    else:
        G = _synthesis_filters(N, min(D, N), T, low_resolution)
    aux = None

    from . import _steering as post

    key = code[0] if code else " "
    if key == "q":
        arr = post.qdht(arr, "inv", code[1:], N, D, *args)
    elif key == "r":
        arr, aux = post.rdht(arr, N, D, "inv", *(args or ("grad",)), return_theta=True)
    elif key == "d":
        arr, aux = post.ddht(arr, N, D, "inv", *(args or ("grad",)), return_theta=True)
    elif key in {"s", "c"}:
        tau = args[0] if args else None
        theta = args[1] if len(args) > 1 else (None if key == "c" else ...)
        if theta is ...:
            arr, aux, _ = post.sdht2(arr, N, D, tau)
        else:
            arr, aux, _ = post.sdht2(arr, N, D, tau, theta)
    elif key == "g":
        offset = ceil(N / T)
        inner = arr[offset:-offset, offset:-offset, 0] if offset else arr[..., 0]
        mean, std = float(np.mean(inner)), float(np.std(inner))
        arr[..., 0] = 0.5 if std == 0 else (1 + np.tanh((arr[..., 0] - mean) / std)) / 2
    elif key == "e":
        from ._misc import edht

        arr = edht(arr, *(args or ()))
    elif key == "m":
        if len(args) < 2:
            raise ValueError("cod='m' requires operation and scale M")
        amount = int(args[1])
        D, N = min(D - amount, N - amount), N - amount
        if min(D, N) < 0:
            raise ValueError("morphological scale exceeds N or D")
        G = T * dhtmtx(N, min(D, N))[::-1, :]
    elif key in {"E", "D"}:
        sign = 1 if key == "E" else -1
        count = min(D, N)
        components = []
        for order in range(1, count + 1):
            a, b = post.gauge(arr, N, D, order, components=True)
            components.append(np.hypot(a, b))
        polynomial = sign * np.poly(np.ones(count))
        for order in range(1, count + 1):
            arr[..., 0] += polynomial[order] * components[order - 1]
    elif key == "p":
        if not args:
            raise ValueError("cod='p' requires the order pair [j,k]")
        horizontal, vertical = map(int, np.asarray(args[0]).ravel()[:2])
        orders = dhtord(N, D, 2)
        selected = np.flatnonzero((orders[:, 0] >= horizontal) & (orders[:, 1] >= vertical))
        if selected.size == 0:
            raise ValueError("higher-order coefficients are required for prediction")
        arr = arr[..., selected].copy()
        D = D - horizontal - vertical
        position = 0
        for total in range(min(D, 2 * N) + 1):
            for y_order in range(max(0, total - N), min(N, total) + 1):
                x_order = total - y_order
                cj = comb(x_order + horizontal, horizontal)
                ck = comb(y_order + vertical, vertical)
                arr[..., position] *= np.sqrt(cj * ck * 0.75**total * 0.25 ** (horizontal + vertical))
                position += 1

    expected = len(dhtord(N, D, 2))
    if arr.shape[2] < expected:
        raise ValueError(f"expected {expected} coefficient channels, got {arr.shape[2]}")
    arr, ysiz, t0, conv_mode = _expand_coefficients_2d(arr, target, N, T, shape)
    rows, cols = np.arange(t0, ysiz[0], T), np.arange(t0, ysiz[1], T)
    if arr.shape[:2] != (len(rows), len(cols)):
        raise ValueError("coefficient dimensions do not match N, T, shape, and xsiz")
    result = np.zeros(target, dtype=float)
    for channel, (horizontal, vertical) in enumerate(dhtord(N, D, 2)):
        up = np.zeros(ysiz, dtype=float)
        up[np.ix_(rows, cols)] = arr[..., channel]
        result += convolve2d(up, np.outer(G[:, vertical], G[:, horizontal]), mode=conv_mode)
    if quincunx:
        result = np.divide(result, quincunx_weight, out=np.zeros_like(result), where=quincunx_weight != 0)
    return (result, aux) if return_aux else result


def dhti2(Y: Array, xsiz: Sequence[int], N: int, D: int, T: int, sumopt: str = "nosum", shape: str = "full", cod: str = " ") -> Array:
    """Interpolate every 2-D coefficient contribution."""

    arr = np.asarray(Y, dtype=float)
    if arr.ndim == 2:
        arr = arr[..., None]
    orders = dhtord(N, D, 2)
    extra = max(0, arr.shape[2] - len(orders))
    pieces = []
    for channel in range(arr.shape[2]):
        source_order = (0, 0) if channel <= extra else tuple(orders[channel - extra])
        one = np.zeros(arr.shape[:2] + (len(orders),), dtype=float)
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
            selected = [pieces[i + extra] for i, order in enumerate(orders) if sum(order) == total and i + extra < len(pieces)]
            result.append(np.sum(selected, axis=0) if selected else np.zeros(tuple(xsiz[:2])))
        return np.stack(result, axis=-1)
    if option == "predict":
        from ._multiscale import pdht

        return pdht(arr, tuple(xsiz[:2]), min(D, N), shape)
    raise ValueError(f"unrecognized sumopt {sumopt!r}")


def dht3(X: Array, N: int, D: int, T: int, cod: str = "", *args, shape: str | None = None) -> Array:
    """Three-dimensional DHT with coefficient channels on the last axis."""

    mode = shape or (cod if len(cod) > 2 else "full")
    code = "" if len(cod) > 2 else cod
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("dht3 expects a 3-D array")
    H = dhtmtx(N, min(D, N))
    work, conv_mode = X, mode
    if mode in {"symm", "repeat", "asymm", "cyclic"}:
        work, conv_mode = _pad_spatial(X, N, mode, 3), "valid"
    maps = []
    for x_order, y_order, z_order in dhtord(N, D, 3):
        kernel = H[:, y_order, None, None] * H[None, :, x_order, None] * H[None, None, :, z_order]
        maps.append(signal_convolve(work, kernel, mode=conv_mode)[::T, ::T, ::T])
    Y = np.stack(maps, axis=-1)
    if code.lower().startswith("r"):
        from ._steering import rdht2

        Y = rdht2(Y, N, D, "fwd", *args)
    return Y


def idht3(Y: Array, xsiz: Sequence[int], N: int, D: int, T: int, cod: str = "", *args, shape: str | None = None) -> Array:
    """Inverse three-dimensional DHT."""

    mode = shape or (cod if len(cod) > 2 else "full")
    code = "" if len(cod) > 2 else cod
    arr = np.asarray(Y, dtype=float)
    if code.lower().startswith("r"):
        from ._steering import rdht2

        arr = rdht2(arr, N, D, "inv", *args)
    target = tuple(int(v) for v in xsiz[:3])
    if mode == "full":
        ysiz, conv_mode = tuple(v + N for v in target), "valid"
    elif mode == "same":
        ysiz, conv_mode = target, "same"
    elif mode == "valid":
        ysiz, conv_mode = tuple(v - N for v in target), "full"
    else:
        raise NotImplementedError("3-D inverse boundary extension supports full/same/valid")
    G = _synthesis_filters(N, min(D, N), T)
    lattice = tuple(np.arange(0, v, T) for v in ysiz)
    expected = tuple(len(v) for v in lattice)
    if arr.shape[:3] != expected:
        raise ValueError(f"expected coefficient grid {expected}, got {arr.shape[:3]}")
    result = np.zeros(target, dtype=float)
    for channel, (x_order, y_order, z_order) in enumerate(dhtord(N, D, 3)):
        up = np.zeros(ysiz, dtype=float)
        up[np.ix_(*lattice)] = arr[..., channel]
        kernel = G[:, y_order, None, None] * G[None, :, x_order, None] * G[None, None, :, z_order]
        result += signal_convolve(up, kernel, mode=conv_mode)
    return result


def _rbt_vector(vector: Array, fnorm: float) -> Array:
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
    return np.concatenate((_rbt_vector(y0, fnorm), _rbt_vector(y1, fnorm)), axis=0)


def fbt(X: Array, N: int, dim: int = 1, K0: float = np.sqrt(2.0)) -> Array:
    """Fast Binomial Transform, blockwise and self-inverse for default K0."""

    arr = np.asarray(X, dtype=float)
    axis = int(dim) - 1
    moved = np.moveaxis(arr, axis, 0)
    T = _integer("N", N) + 1
    length = (moved.shape[0] // T) * T
    moved = moved[:length]
    flat = moved.reshape(length, -1)
    out = np.empty_like(flat)
    for start in range(0, length, T):
        out[start : start + T] = _rbt_vector(flat[start : start + T], float(K0))
    return np.moveaxis(out.reshape((length,) + moved.shape[1:]), 0, axis)


def fbt2(X: Array, N: int) -> Array:
    return fbt(fbt(X, N, 2), N, 1)


def fbt3(X: Array, N: int) -> Array:
    return fbt(fbt(fbt(X, N, 3), N, 2), N, 1)


def fbtmtx(N: int) -> Array:
    B = dhtmtx(N)
    scale = (2.0**N) / np.sqrt((2.0**N) * B[:, 0])
    return np.rint(B * scale[None, :])


# The original dhtJ.m is a lightly modified duplicate of dht.m.
dhtJ = dht
