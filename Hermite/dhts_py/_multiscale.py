"""Multiscale DHT, pyramids, prediction, and scale-space applications."""

from __future__ import annotations

from math import comb, floor, log2
from typing import Sequence

import numpy as np

from ._core import dht, dht2, dhti, dhti2, dhtord, idht, idht2
from ._steering import ddht, gauge, rdht, sdht2


Array = np.ndarray


def _scale_degree(value, default_n: int) -> tuple[int, int]:
    values = np.atleast_1d(value).astype(int)
    return (int(values[0]), int(values[1])) if values.size > 1 else (default_n, int(values[0]))


def _spatial_shape(coefficients: Array, ndim: int) -> tuple[int, ...]:
    shape = list(coefficients.shape[:ndim])
    return tuple(shape)


def binpyr(X: Array, M, dim: int | None = None, shape: str = "full") -> list[Array]:
    """One-dimensional binomial pyramid."""

    values = np.atleast_1d(M).astype(int)
    levels = int(values[0])
    N = int(values[1]) if values.size > 1 else 8
    T = int(round(np.sqrt(N / 2)))
    result = [dht(X, N, 0, T, dim, shape)]
    # binpyr.m stores Y{1} and then appends Y{m+1} for m=1:M.
    for _ in range(levels):
        # Remove the singleton coefficient axis before the next level.
        result.append(2 * dht(np.squeeze(result[-1], axis=-1), 6, 0, 2, dim, shape))
    return result


def binpyr2(X: Array, M, shape: str = "full") -> list[Array]:
    """Two-dimensional binomial pyramid."""

    values = np.atleast_1d(M).astype(int)
    levels = int(values[0])
    N = int(values[1]) if values.size > 1 else 8
    result = [np.squeeze(dht2(X, N, 0, int(round(np.sqrt(N / 2))), shape), axis=2)]
    for _ in range(1, levels):
        result.append(np.squeeze(dht2(result[-1], 6, 0, 2, shape), axis=2))
    return result


def mdht(X: Array, D, M: int, dim: int | None = None, cod: str = " ", *args, shape: str | None = None) -> list[Array]:
    """One-dimensional multiscale DHT."""

    if shape is None and len(cod) > 3:
        shape, cod = cod.lower(), (str(args[0]) if args else " ")
        args = args[1:] if args else ()
    shape = shape or "full"
    N, degree = _scale_degree(D, 8)
    T = int(round(np.sqrt(N / 2)))
    current = np.asarray(X, dtype=float)
    result: list[Array] = []
    for level in range(int(M)):
        prefix = "h" if level == 0 else "l"
        if level == M - 1:
            result.append(dht(current, N, degree, T, dim, shape, prefix + cod, *args))
        else:
            high, current = dht(current, N, degree, T, dim, shape, prefix + cod, *args, return_lowpass=True)
            result.append(high)
        N, T = 6, 2
    return result


def imdht(Y: Sequence[Array], xsiz: Sequence[int] | int, D, dim: int | None = None, cod: str = " ", *args, shape: str | None = None) -> Array:
    """Inverse one-dimensional multiscale DHT."""

    if shape is None and len(cod) > 3:
        shape, cod = cod.lower(), (str(args[0]) if args else " ")
        args = args[1:] if args else ()
    shape = shape or "full"
    N0, degree = _scale_degree(D, 8)
    T0 = int(round(np.sqrt(N0 / 2)))
    target = tuple(np.atleast_1d(xsiz).astype(int))
    if dim is None:
        dim = next((i + 1 for i, v in enumerate(target) if v > 1), 1)
    current = np.asarray(Y[-1], dtype=float)
    for level in range(len(Y) - 2, -1, -1):
        saved = np.asarray(Y[level], dtype=float)
        stage_shape = list(target)
        stage_shape[dim - 1] = saved.shape[dim - 1]
        low = idht(current, stage_shape, 6, degree, 2, dim, "l" + cod, *args, shape=shape)
        current = low if saved.size == 0 else np.concatenate((low[..., None], saved), axis=-1)
    return idht(current, target, N0, degree, T0, dim, "h" + cod, *args, shape=shape)


def mdhti(Y: Sequence[Array], xsiz: Sequence[int] | int, D, dim: int | None = None, sumopt: str = "nosum", cod: str = " ", *args, shape: str | None = None) -> Array:
    """Interpolate a one-dimensional multiscale decomposition."""

    if shape is None and len(cod) > 3:
        shape, cod = cod.lower(), (str(args[0]) if args else " ")
        args = args[1:] if args else ()
    shape = shape or "full"
    N0, degree = _scale_degree(D, 8)
    T0 = int(round(np.sqrt(N0 / 2)))
    target = tuple(np.atleast_1d(xsiz).astype(int))
    if dim is None:
        dim = next((i + 1 for i, v in enumerate(target) if v > 1), 1)
    current = np.asarray(Y[-1], dtype=float)
    for level in range(len(Y) - 2, -1, -1):
        saved = np.asarray(Y[level], dtype=float)
        stage_shape = list(target)
        stage_shape[dim - 1] = saved.shape[dim - 1]
        interpolated = dhti(current, stage_shape, 6, degree, 2, dim, sumopt, "l" + cod, *args, shape=shape)
        current = interpolated if saved.size == 0 else np.concatenate((interpolated, saved), axis=-1)
    return dhti(current, target, N0, degree, T0, dim, sumopt, "h" + cod, *args, shape=shape)


def mdht2(X: Array, D, M: int, cod: str = "", *args, shape: str | None = None, return_aux: bool = False):
    """Two-dimensional multiscale DHT."""

    if shape is None and len(cod) > 3:
        shape, cod = cod.lower(), (str(args[0]) if args else "")
        args = args[1:] if args else ()
    shape = shape or "full"
    N, degree = _scale_degree(D, 8)
    T = int(round(np.sqrt(N / 2)))
    predict = "p" in cod
    keep = cod.startswith("0")
    post_code = cod.replace("p", "")
    if post_code.startswith("0"):
        post_code = post_code[1:]
    current = np.asarray(X, dtype=float)
    result: list[Array] = []
    auxiliary: list = []
    for level in range(int(M)):
        prefix = "h" if level == 0 else ("L" if level == M - 1 else "l")
        call_args = [v[level] if isinstance(v, (list, tuple)) and len(v) == M else v for v in args]
        transformed = dht2(current, N, degree, T, shape, prefix + post_code, *call_args, return_aux=return_aux)
        if return_aux:
            transformed, aux = transformed
            auxiliary.append(aux)
        if level == M - 1 or keep:
            result.append(transformed)
        else:
            result.append(transformed[..., 1:, :] if transformed.ndim == 4 else transformed[..., 1:])
        current = transformed[..., 0, :] if transformed.ndim == 4 else transformed[..., 0]
        N, T = 6, 2
    if predict:
        result = pdht(result, degree, "fwd", shape)
    return (result, auxiliary) if return_aux else result


def imdht2(Y: Sequence[Array], xsiz: Sequence[int], D, cod: str = "", *args, shape: str | None = None) -> Array:
    """Inverse two-dimensional multiscale DHT."""

    if shape is None and len(cod) > 3:
        shape, cod = cod.lower(), (str(args[0]) if args else "")
        args = args[1:] if args else ()
    shape = shape or "full"
    N0, degree = _scale_degree(D, 8)
    T0 = int(round(np.sqrt(N0 / 2)))
    keep = cod.startswith("0")
    post_code = cod[1:] if keep else cod
    levels = list(Y)
    if post_code.startswith("p"):
        levels = pdht(levels, degree, "inv", shape, post_code[1:], *args)
        post_code = "q" if "q" in post_code else ""
    current = np.asarray(levels[-1], dtype=float)
    for level in range(len(levels) - 2, -1, -1):
        saved = np.asarray(levels[level], dtype=float)
        stage_shape = saved.shape[:2]
        call_args = [v[level + 1] if isinstance(v, (list, tuple)) and len(v) == len(levels) else v for v in args]
        low = idht2(current, stage_shape, 6, degree, 2, shape, "l" + post_code, *call_args)
        if low.ndim == 2:
            low = low[..., None]
        current = np.concatenate((low, saved), axis=2)
    call_args = [v[0] if isinstance(v, (list, tuple)) and len(v) == len(levels) else v for v in args]
    return idht2(current, xsiz, N0, degree, T0, shape, "h" + post_code, *call_args)


def mdhti2(Y: Sequence[Array], xsiz: Sequence[int], D, sumopt: str = "nosum", shape: str = "full") -> Array:
    """Interpolate a 2-D multiscale decomposition."""

    N0, degree = _scale_degree(D, 8)
    T0 = int(round(np.sqrt(N0 / 2)))
    current = np.asarray(Y[-1], dtype=float)
    for level in range(len(Y) - 2, -1, -1):
        saved = np.asarray(Y[level], dtype=float)
        current = np.concatenate((dhti2(current, saved.shape[:2], 6, degree, 2, sumopt, shape, "l"), saved), axis=2)
    return dhti2(current, xsiz, N0, degree, T0, sumopt, shape, "h")


def mrdht(Y: Sequence[Array], D, tdir: str = "fwd", theta="grad", *, return_theta: bool = False):
    N, degree = _scale_degree(D, 8)
    angles = [theta] * len(Y) if isinstance(theta, str) or np.isscalar(theta) else list(theta)
    result, used = [], []
    for level, saved in enumerate(Y):
        if level:
            N = 6
        arr = np.asarray(saved, dtype=float)
        work = arr if level == len(Y) - 1 else np.concatenate((np.ones(arr.shape[:2] + (1,)), arr), axis=2)
        transformed, angle = rdht(work, N, degree, tdir, angles[level], return_theta=True)
        result.append(transformed if level == len(Y) - 1 else transformed[..., 1:])
        used.append(angle)
    return (result, used) if return_theta else result


def mrdht2(Y: Sequence[Array], D, *args) -> list[Array]:
    N, degree = _scale_degree(D, 2)
    result = []
    for level, saved in enumerate(Y):
        if level == 1:
            N = 6
        arr = np.asarray(saved, dtype=float)
        work = arr if level == len(Y) - 1 else np.concatenate((np.zeros(arr.shape[:2] + (1,)), arr), axis=2)
        transformed = rdht(work, N, degree, *args)
        result.append(transformed if level == len(Y) - 1 else transformed[..., 1:])
    return result


def mddht2(Y: Sequence[Array], D, *args) -> list[Array]:
    N, degree = _scale_degree(D, 2)
    result = []
    for level, saved in enumerate(Y):
        if level == 1:
            N = 6
        arr = np.asarray(saved, dtype=float)
        work = arr if level == len(Y) - 1 else np.concatenate((np.zeros(arr.shape[:2] + (1,)), arr), axis=2)
        transformed = ddht(work, N, degree, *args)
        result.append(transformed if level == len(Y) - 1 else transformed[..., 1:])
    return result


def mgauge(Y: Sequence[Array], D, L=1, *, components: bool = False):
    N, degree = _scale_degree(D, 8)
    first, second = [], []
    for level, saved in enumerate(Y):
        if level:
            N = 6
        arr = np.asarray(saved, dtype=float)
        work = arr if level == len(Y) - 1 else np.concatenate((np.zeros(arr.shape[:2] + (1,)), arr), axis=2)
        if components:
            a, b = gauge(work, N, degree, L, components=True)
            first.append(a)
            second.append(b)
        else:
            first.append(gauge(work, N, degree, L))
    return (first, second) if components else first


def _predict_single(Y: Array, ysiz: Sequence[int], D: int, shape: str, cod: str = "", *args) -> Array:
    N, T, tau = 6, 2, 0.75
    degree = min(int(D), N)
    orders = dhtord(N, degree, 2)
    source = np.asarray(Y, dtype=float)
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
                    cx = comb(n - m + horizontal, n - m)
                    cy = comb(m + vertical, m)
                    z[..., position] *= np.sqrt(cx * cy * (1 - tau) ** total)
                    position += 1
            predictions.append(idht2(z, ysiz, N, remaining, T, shape, cod, *args))
    return np.stack(predictions, axis=-1)


def pdht(Y, D, tdir="fwd", cod: str = "", *args, shape: str | None = None):
    """Predict 2-D DHT coefficients or multiscale prediction residues."""

    if not isinstance(Y, (list, tuple)):
        ysiz = D
        degree = int(tdir)
        mode = shape or (cod if len(cod) > 3 else "full")
        post = "" if len(cod) > 3 else cod
        return _predict_single(np.asarray(Y), ysiz, degree, mode, post, *args)
    mode = shape or (cod if len(cod) > 3 else "full")
    code = "" if len(cod) > 3 else cod
    result = [np.asarray(v, dtype=float).copy() for v in Y]
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


def bsmooth(X: Array, N: int) -> Array:
    """Approximate 1-D binomial smoothing through a multiscale DHT."""

    M = floor(log2(np.sqrt(N))) - 1
    if M <= 0:
        return np.squeeze(dht(X, N, 0, 1, shape="same"), axis=-1)
    tau = (4 ** (M + 1) - N) / (3 * 4**M)
    pyramid = mdht(X, 0, M)
    last = np.asarray(pyramid[-1])
    coeff = dht(np.squeeze(last[..., 0] if last.shape[-1] == 1 else last), 6, 6, 2)
    coeff *= tau ** np.arange(7)
    pyramid[-1] = idht(coeff, np.squeeze(last).shape, 6, 6, 2)
    return imdht(pyramid, np.asarray(X).shape, 0)


def bsmooth2(X: Array, S, maxgrad=None) -> Array:
    """Approximate 2-D binomial smoothing at one or more scales."""

    scales = np.atleast_1d(S).astype(float)
    k = np.ceil(np.log2(scales) / 2 - 1).astype(int)
    tau = (4 - scales / (4.0**k)) / 3.5
    M = int(k.max()) + 1
    code = "0E" if maxgrad is not None else "0"
    pyramid = mdht2(X, [2, 12], M, "repeat", code, *(() if maxgrad is None else (maxgrad,)))
    stage: list[Array] = []
    for level in range(M):
        indices = np.flatnonzero(k == level)
        target = np.asarray(X).shape[:2] if level == 0 else np.asarray(pyramid[level - 1]).shape[:2]
        reconstructed = []
        for index in indices[::-1]:
            if tau[index] != 0:
                reconstructed.append(idht2(pyramid[level], target, 2 if level == 0 else 6, 12, 1 if level == 0 else 2, "repeat", "s", tau[index]))
            else:
                reconstructed.append(idht2(np.asarray(pyramid[level])[..., :1], target, 2 if level == 0 else 6, 0, 1 if level == 0 else 2, "repeat"))
        stage.append(np.stack(reconstructed, axis=-1) if reconstructed else np.empty(target + (0,)))
    if M == 1:
        return stage[0]
    upper = mdhti2(stage[1:], np.asarray(X).shape[:2], [2, 0], "nosum", "repeat")
    return np.concatenate((stage[0], upper[..., ::-1]), axis=-1)


def lorient(X: Array, M: int | None = None):
    """Multiscale local orientation estimate."""

    image = np.asarray(X, dtype=float)
    levels = floor(log2(min(image.shape)) / 2) if M is None else int(M)
    coeff = mdhti2(mdht2(image, 1, levels), image.shape, 0)
    xx_ch = coeff[..., 1::2]
    yy_ch = coeff[..., 2::2]
    count = min(xx_ch.shape[-1], yy_ch.shape[-1])
    yy = np.sum(2 * xx_ch[..., :count] * yy_ch[..., :count], axis=-1)
    xx = np.sum(xx_ch[..., :count] ** 2 - yy_ch[..., :count] ** 2, axis=-1)
    return 0.5 * np.arctan2(yy, xx) + np.pi / 2, np.hypot(xx, yy)


def mscode(X: Array, S=None, Tol: float = 0, *args):
    """Multiscale comparison code and most-significant transition layer."""

    image = np.asarray(X, dtype=float)
    if S is None:
        levels = min(floor(log2(v)) for v in image.shape[:2])
        scales = 4.0 ** np.arange(1, levels + 1)
    else:
        scales = np.atleast_1d(S)
    smooth = bsmooth2(image + Tol, scales, *args)
    code = np.zeros(image.shape, dtype=np.uint64)
    transition = np.ones(image.shape, dtype=int)
    previous = np.ones(image.shape, dtype=bool)
    for n in range(len(scales) - 1, -1, -1):
        bit = image >= smooth[..., n]
        code |= bit.astype(np.uint64) << n
        transition[(transition == 1) & previous & ~bit] = n + 2
        previous = bit
    transition[transition == len(scales) + 1] = 0
    stack = np.concatenate((image[..., None], smooth), axis=-1)
    selected = np.take_along_axis(stack, transition[..., None], axis=-1)[..., 0]
    return code, transition, selected


def overshoot(X: Array, M: int) -> Array:
    """Enhance edge transitions using the multiscale construction in overshoot.m."""

    image = np.asarray(X, dtype=float)
    Y0 = dht2(image, 2, 4, 1)
    alpha = 4.0
    tau = alpha / (alpha - 1) - 1 / np.log(alpha)
    padded = np.zeros(tuple(v + 2 for v in image.shape))
    padded[1:-1, 1:-1] = idht2(Y0, image.shape, 2, 4, 1, "s", tau)
    base = padded
    high = mdht2(np.log(4) * base, [6, 1], M)
    low = mdht2(Y0[..., 0], [6, 0], M)
    high[-1][..., 0] = low[-1][..., 0]
    Y0[..., 0] = imdht2(high, Y0.shape[:2], [6, 1])
    return idht2(Y0, image.shape, 2, 4, 1)
