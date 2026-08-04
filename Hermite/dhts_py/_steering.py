"""Steering, gauge, smoothing, morphology, and DHT invariants."""

from __future__ import annotations

from math import comb
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import convolve2d

from ._core import dhtord


Array = np.ndarray
_MISSING = object()


def _binomial_norm(degree: int) -> Array:
    return np.sqrt(np.asarray([comb(degree, k) for k in range(degree + 1)], dtype=float))


def _theta_map(theta, shape: tuple[int, ...]) -> Array:
    value = np.asarray(theta, dtype=float)
    if value.ndim == 0:
        return np.full(shape, float(value), dtype=float)
    if value.shape != shape:
        try:
            return np.broadcast_to(value, shape).astype(float, copy=False)
        except ValueError as exc:
            raise ValueError(f"theta has shape {value.shape}; expected {shape}") from exc
    return value


def _channel_blocks(N: int, D: int) -> list[tuple[int, int]]:
    """Return ``(start, degree)`` blocks used by RDHT."""

    blocks: list[tuple[int, int]] = []
    start = 1
    for total in range(1, min(D, N) + 1):
        blocks.append((start, total))
        start += total + 1
    for offset in range(1, min(D - N, N - 1) + 1):
        degree = N - offset
        blocks.append((start, degree))
        start += degree + 1
    return blocks


def _rotate_block(block: Array, theta: Array) -> Array:
    """Normalized RDHT recurrence for one complete homogeneous block."""

    degree = block.shape[-1] - 1
    if degree <= 0:
        return block.copy()
    c, s = np.cos(theta), np.sin(theta)
    C = _binomial_norm(degree)
    h = np.asarray(block, dtype=float).copy()
    if degree > 1:
        h[..., 1:degree] /= C[1:degree]
    out = np.empty_like(h)
    hlen = degree + 1
    for m in range(degree):
        low = h.copy()
        llen = hlen
        for _ in range(m, degree):
            low = c[..., None] * low[..., : llen - 1] + s[..., None] * low[..., 1:llen]
            llen -= 1
        out[..., m] = low[..., 0] * C[m]
        h = c[..., None] * h[..., 1:hlen] - s[..., None] * h[..., : hlen - 1]
        hlen -= 1
    out[..., degree] = h[..., 0]
    return out


def gauge(Y: Array, N: int, D: int, L: int | str = 1, *, components: bool = False, magnitude: bool = False):
    """Compute the general gauge condition from ``gauge.m``.

    The default result is an orientation map.  ``components=True`` returns
    the two gauge components; ``magnitude=True`` returns angle and magnitude.
    """

    if isinstance(L, str):
        key = L.lower()
        L = 1 if key == "grad" else 2 if key == "hess" else int(key)
    L = int(L)
    if not 1 <= L <= D:
        raise ValueError("L must satisfy 1 <= L <= D")
    arr = np.asarray(Y, dtype=float)
    if arr.ndim == 4:
        values = [gauge(arr[..., k], N, D, L, components=components, magnitude=magnitude) for k in range(arr.shape[3])]
        if isinstance(values[0], tuple):
            return tuple(np.stack([v[i] for v in values], axis=-1) for i in range(len(values[0])))
        return np.stack(values, axis=-1)
    if arr.ndim != 3:
        raise ValueError("gauge expects a coefficient stack")

    orders = dhtord(N, D, 2)
    selected = np.flatnonzero(orders.sum(axis=1) == L)
    if selected.size == 0:
        raise ValueError(f"coefficient order {L} is not available")
    degree = selected.size - 1
    block = arr[..., selected]
    C = _binomial_norm(degree)
    odd_weights = C[1::2] * (-1.0) ** np.arange(C[1::2].size)
    even_weights = C[0::2] * (-1.0) ** np.arange(C[0::2].size)
    a = np.tensordot(block[..., 1::2], odd_weights, axes=([-1], [0]))
    b = np.tensordot(block[..., 0::2], even_weights, axes=([-1], [0]))
    if components:
        return a, b
    theta = np.arctan2(a, b) / max(degree, 1)
    if degree > 1:
        increments = np.arange(degree + 1) * np.pi / degree
        responses = np.stack(
            [
                np.sum(
                    block
                    * C.reshape((1,) * (block.ndim - 1) + (-1,))
                    * np.cos(theta[..., None] + inc) ** (degree - np.arange(degree + 1))
                    * np.sin(theta[..., None] + inc) ** np.arange(degree + 1),
                    axis=-1,
                )
                for inc in increments
            ],
            axis=-1,
        )
        theta = theta + np.argmax(np.abs(responses), axis=-1) * (np.pi / degree)
        if arr.shape[-1] >= 3:
            sign = np.cos(theta) * arr[..., 1] + np.sin(theta) * arr[..., 2]
            theta = np.where(sign < 0, theta - np.pi, theta)
    return (theta, np.hypot(a, b)) if magnitude else theta


def rdht(Y: Array, N: int, D: int, tdir: str = "fwd", theta=None, *, return_theta: bool = False):
    """Rotate normalized 2-D DHT coefficients (translation of ``rdht.m``)."""

    arr = np.asarray(Y, dtype=float)
    if arr.ndim == 4:
        values = [rdht(arr[..., k], N, D, tdir, theta, return_theta=return_theta) for k in range(arr.shape[3])]
        if return_theta:
            return np.stack([v[0] for v in values], axis=-1), values[0][1]
        return np.stack(values, axis=-1)
    if arr.ndim < 3:
        return (arr.copy(), theta) if return_theta else arr.copy()
    direction = tdir.lower()
    gauge_condition = "grad" if theta is None else (theta.lower() if isinstance(theta, str) else "none")
    work = arr.copy()
    null_channel: int | None = None

    if direction == "inv":
        if gauge_condition in {"grad", "grad180"}:
            angle = -2 * np.pi * work[..., 2]
            work[..., 2] = 0
        elif gauge_condition == "hess":
            angle = -2 * np.pi * work[..., 4]
            work[..., 4] = 0
        elif gauge_condition == "none":
            angle = -_theta_map(theta, work.shape[:2])
        else:
            angle = -2 * np.pi * work[..., -1]
            work = work[..., :-1]
    elif direction == "fwd":
        if gauge_condition == "grad":
            angle, null_channel = gauge(work, N, D, 1), 2
        elif gauge_condition == "grad180":
            angle, null_channel = gauge(work, N, D, 1) + np.pi, 2
        elif gauge_condition == "hess":
            angle, null_channel = gauge(work, N, D, 2), 4
        elif gauge_condition == "none":
            angle = _theta_map(theta, work.shape[:2])
        else:
            angle = gauge(work, N, D, int(gauge_condition))
            if not return_theta:
                null_channel = work.shape[-1]
                work = np.concatenate((work, np.zeros(work.shape[:2] + (1,))), axis=-1)
    else:
        raise ValueError("tdir must be 'fwd' or 'inv'")

    out = work.copy()
    for start, degree in _channel_blocks(N, D):
        stop = start + degree + 1
        if stop <= work.shape[-1]:
            out[..., start:stop] = _rotate_block(work[..., start:stop], angle)
    if null_channel is not None:
        out[..., null_channel] = angle / (2 * np.pi)
    return (out, angle if direction == "fwd" else -angle) if return_theta else out


def ddht(Y: Array, N: int, D: int, tdir: str = "fwd", theta=None, *, return_theta: bool = False):
    """Directional Hermite coefficients from ``ddht.m``."""

    arr = np.asarray(Y, dtype=float)
    if arr.ndim == 4:
        values = [ddht(arr[..., k], N, D, tdir, theta, return_theta=return_theta) for k in range(arr.shape[3])]
        if return_theta:
            return np.stack([v[0] for v in values], axis=-1), values[0][1]
        return np.stack(values, axis=-1)
    condition = "grad" if theta is None else (theta.lower() if isinstance(theta, str) else "none")
    work = arr.copy()
    null_channel: int | None = None
    if tdir.lower() == "fwd":
        if condition == "grad":
            angle, null_channel = gauge(work, N, D, 1), 2
        elif condition == "hess":
            angle, null_channel = gauge(work, N, D, 2), 4
        elif condition == "none":
            angle = _theta_map(theta, work.shape[:2])
        else:
            angle = gauge(work, N, D, int(condition))
            if not return_theta:
                null_channel = work.shape[-1]
                work = np.concatenate((work, np.zeros(work.shape[:2] + (1,))), axis=-1)
        out = work.copy()
        for start, degree in _channel_blocks(N, D):
            C = _binomial_norm(degree)
            powers = np.arange(degree + 1)
            block = work[..., start : start + degree + 1]
            for j in range(degree + 1):
                a = angle + j * np.pi / (degree + 1)
                out[..., start + j] = np.sum(
                    block * C * np.cos(a)[..., None] ** (degree - powers) * np.sin(a)[..., None] ** powers,
                    axis=-1,
                )
        if null_channel is not None:
            out[..., null_channel] = angle / (2 * np.pi)
    elif tdir.lower() == "inv":
        if condition == "grad":
            angle = 2 * np.pi * work[..., 2]
            work[..., 2] = 0
        elif condition == "hess":
            angle = 2 * np.pi * work[..., 4]
            work[..., 4] = 0
        elif condition == "none":
            angle = _theta_map(theta, work.shape[:2])
        else:
            angle = 2 * np.pi * work[..., -1]
            work = work[..., :-1]
        out = work.copy()
        for start, degree in _channel_blocks(N, D):
            C = _binomial_norm(degree)
            powers = np.arange(degree + 1)
            directions = angle[..., None] + np.arange(degree + 1) * np.pi / (degree + 1)
            basis = np.cos(directions)[..., None] ** (degree - powers) * np.sin(directions)[..., None] ** powers
            norm = np.sum(np.sin(np.arange(degree + 1) * np.pi / (degree + 1)) ** (2 * degree))
            for m in range(degree + 1):
                out[..., start + m] = C[m] * np.sum(work[..., start : start + degree + 1] * basis[..., m], axis=-1) / norm
    else:
        raise ValueError("tdir must be 'fwd' or 'inv'")
    return (out, angle) if return_theta else out


def _rotate_tensor_pair(Y: Array, N: int, D: int, axes: tuple[int, int], theta: Array) -> Array:
    orders = dhtord(N, D, 3)
    out = Y.copy()
    fixed_axis = ({0, 1, 2} - set(axes)).pop()
    for fixed in range(N + 1):
        indices_fixed = np.flatnonzero(orders[:, fixed_axis] == fixed)
        totals = np.unique(orders[indices_fixed][:, axes].sum(axis=1))
        for total in totals:
            indices = indices_fixed[orders[indices_fixed][:, axes].sum(axis=1) == total]
            indices = indices[np.argsort(orders[indices, axes[0]])]
            if indices.size > 1:
                out[..., indices] = _rotate_block(Y[..., indices], theta)
    return out


def rdht2(Y: Array, N: int, D: int, flag: str = "fwd", theta=None, phi=None) -> Array:
    """Rotate 3-D coefficient tensors with two Euler-plane rotations."""

    arr = np.asarray(Y, dtype=float)
    if arr.ndim != 4:
        raise ValueError("rdht2 expects (rows, columns, depth, coefficients)")
    if theta is None:
        theta = np.arctan2(arr[..., 2], arr[..., 1])
    if phi is None:
        phi = np.zeros(arr.shape[:3], dtype=float)
    theta = _theta_map(theta, arr.shape[:3])
    phi = _theta_map(phi, arr.shape[:3])
    if flag.lower() == "fwd":
        return _rotate_tensor_pair(_rotate_tensor_pair(arr, N, D, (0, 1), theta), N, D, (0, 2), phi)
    if flag.lower() == "inv":
        return _rotate_tensor_pair(_rotate_tensor_pair(arr, N, D, (0, 2), -phi), N, D, (0, 1), -theta)
    raise ValueError("flag must be 'fwd' or 'inv'")


def energy(Y, N: int, D: int | str | None = None, cod: str = "ac"):
    """Coefficient-based energies from ``energy.m``."""

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
    arr = np.asarray(Y, dtype=float)
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


def _perceptual_tau(rotated: Array, N: int, D: int) -> Array:
    """Deterministic fallback for the unavailable pdcentr.mat classifier."""

    total = energy(rotated, N, D, "ac")
    one_d = energy(rotated, N, D, "1d")
    ratio = one_d / (total + np.finfo(float).eps)
    active = total > np.nanmedian(total)
    label_1d = active & (ratio >= 0.75)
    label_2d = active & ~label_1d
    return np.stack((label_1d | label_2d, label_2d), axis=-1).astype(float)


def sdht2(Y: Array, N: int, D: int, tau=None, theta=_MISSING):
    """Gaussian smoothing of DHT coefficients.

    Omitting ``theta`` selects isotropic smoothing.  Passing ``theta=None``
    selects the adaptive anisotropic branch, as in MATLAB ``sdht2(...,[],[])``.
    """

    arr = np.asarray(Y, dtype=float).copy()
    if arr.ndim == 4:
        bands = [sdht2(arr[..., k], N, D, tau, theta) for k in range(arr.shape[3])]
        return np.stack([b[0] for b in bands], axis=-1), bands[0][1], bands[0][2]
    if theta is _MISSING:
        if tau is None:
            first = arr[..., 1] ** 2 + arr[..., 2] ** 2
            std = np.std(first)
            tau = np.full(first.shape, 0.5) if std == 0 else (1 + np.tanh(first / std - 1)) / 2
        elif callable(tau):
            tau = tau(arr)
        tau_map = np.asarray(tau, dtype=float)
        for channel, order in enumerate(dhtord(N, D, 2)):
            if channel:
                arr[..., channel] *= tau_map ** int(order.sum())
        return arr, tau_map, None

    if theta is None:
        angle = np.arctan2(arr[..., 2], arr[..., 1]) if tau is None else np.arctan2(arr[..., 1], -arr[..., 2])
    elif callable(theta):
        angle = theta(arr)
    else:
        angle = _theta_map(theta, arr.shape[:2])
    rotated = rdht(arr, N, D, "fwd", angle)
    tau_map = _perceptual_tau(rotated, N, D) if tau is None else np.asarray(tau, dtype=float)
    if tau_map.ndim == 2:
        tau_map = tau_map[..., None]
    for channel, (horizontal, vertical) in enumerate(dhtord(N, D, 2)):
        if channel:
            factor = tau_map[..., 0] ** horizontal
            if tau_map.shape[-1] > 1:
                factor *= tau_map[..., 1] ** vertical
            rotated[..., channel] *= factor
    return rdht(rotated, N, D, "inv", angle), tau_map, angle


def qdht(Y: Array, tdir: str = "fwd", cod: str = "", N: int | None = None, D: int | None = None, *args) -> Array:
    """Apply coefficient processing on the toolbox quincunx lattice."""

    out = np.asarray(Y, dtype=float).copy()
    odd_rows, odd_cols = np.arange(0, out.shape[0], 2), np.arange(0, out.shape[1], 2)
    even_rows, even_cols = np.arange(1, out.shape[0], 2), np.arange(1, out.shape[1], 2)
    rejected_a = np.ix_(even_rows, odd_cols)
    rejected_b = np.ix_(odd_rows, even_cols)
    if tdir.lower() == "fwd":
        start = 1 if any(ch in cod for ch in "lhL") else 0
        out[rejected_a + (slice(start, None),)] = np.nan
        out[rejected_b + (slice(start, None),)] = np.nan
    elif tdir.lower() == "inv":
        out[rejected_a + (slice(None),)] = 0
        out[rejected_b + (slice(None),)] = 0
    else:
        raise ValueError("tdir must be 'fwd' or 'inv'")
    if not cod:
        return out
    if N is None or D is None:
        raise ValueError("N and D are required when cod is supplied")
    key = cod[0].lower()
    for rows, cols in ((odd_rows, odd_cols), (even_rows, even_cols)):
        sub = out[np.ix_(rows, cols, np.arange(out.shape[2]))]
        local_args = list(args)
        for i, value in enumerate(local_args):
            if isinstance(value, np.ndarray) and value.shape[:2] == out.shape[:2]:
                local_args[i] = value[np.ix_(rows, cols)]
        if key == "r":
            sub = rdht(sub, N, D, tdir, *(local_args or [None]))
        elif key == "d":
            sub = ddht(sub, N, D, tdir, *(local_args or [None]))
        elif key in {"s", "c"}:
            sub = sdht2(sub, N, D, *(local_args if key == "s" else [None, None]))[0]
        out[np.ix_(rows, cols, np.arange(sub.shape[2]))] = sub
    return out


def xdht2(Y: Array, N: int, D: int, M, msk=None, theta=None, *args):
    """Scale-space shift of DHT2 coefficients."""

    arr = np.asarray(Y, dtype=float)
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


def _dhtshift(Y: Array, N: int, D: int, M: int, theta: Array) -> Array:
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


def dhtmorph(Y: Array, N: int, D: int, oper: str, M: int | None = None, theta=None):
    """Morphological erosion/dilation/opening/closing in coefficient space."""

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


_DEFAULT_INVARIANTS = (
    "gauge1", "gauge2", "gauge3", "gauge4", "dflatness", "cornerness",
    "res1d", "umblicity", "laplacian", "flowlinec", "meancurv",
)


def dhtgi(Y: Array, N, D=None, invr_name: str | Iterable[str] | None = None):
    """Compute geometric invariants from DHT coefficients."""

    if isinstance(Y, (list, tuple)):
        scale = np.atleast_1d(N)
        if scale.size > 1:
            base_n, degree = int(scale[0]), int(scale[1])
        else:
            base_n, degree = 8, int(scale[0])
        names_arg = D if D is not None else invr_name
        results = []
        for level, values in enumerate(Y):
            arr_level = np.asarray(values, dtype=float)
            stack = arr_level if level == len(Y) - 1 else np.concatenate((np.zeros(arr_level.shape[:2] + (1,)), arr_level), axis=2)
            results.append(dhtgi(stack, base_n if level == 0 else 6, degree, names_arg)[0])
        return results, list(_DEFAULT_INVARIANTS if names_arg is None else ([names_arg] if isinstance(names_arg, str) else names_arg))
    if D is None:
        raise ValueError("single-scale dhtgi requires D")
    N, D = int(N), int(D)
    names = list(_DEFAULT_INVARIANTS if invr_name is None else ([invr_name] if isinstance(invr_name, str) else invr_name))
    arr = np.asarray(Y, dtype=float)
    values = []
    g = np.hypot(arr[..., 1], arr[..., 2]) if D >= 1 else None
    if D >= 2:
        trace = arr[..., 3] + arr[..., 5]
        hnorm = arr[..., 3] ** 2 + 2 * arr[..., 4] ** 2 + arr[..., 5] ** 2
    for name in names:
        key = name.lower()
        if key == "gradient":
            value = np.sqrt(N) * g
        elif key == "gauge1":
            value = g
        elif key == "laplacian":
            value = trace
        elif key == "dflatness":
            value = np.sqrt(hnorm)
        elif key in {"cornerness", "isophotec"}:
            directional = arr[..., 1] ** 2 * arr[..., 3] + 2 * arr[..., 1] * arr[..., 2] * arr[..., 4] + arr[..., 2] ** 2 * arr[..., 5]
            raw = directional - g * trace
            value = raw if key == "cornerness" else raw / (np.finfo(float).eps + g) ** 1.5
        elif key == "flowlinec":
            value = (arr[..., 4] * (arr[..., 1] ** 2 - arr[..., 2] ** 2) - arr[..., 1] * arr[..., 2] * (arr[..., 4] - arr[..., 3])) / (np.finfo(float).eps + g) ** 1.5
        elif key == "umblicity":
            value = (trace**2 - hnorm) / (np.finfo(float).eps + hnorm)
        elif key == "gaussianc":
            value = (trace**2 - hnorm) / (1 / 255 + g) ** 3
        elif key == "meancurv":
            value = 0.5 * trace / (1 / 255 + g) ** 1.5
        elif key in {"gauge2", "diaghess"}:
            value = np.sqrt((arr[..., 3] - arr[..., 5]) ** 2 + 2 * arr[..., 4] ** 2)
        elif key == "gauge3":
            value = np.sqrt((arr[..., 6] - np.sqrt(3) * arr[..., 8]) ** 2 + (np.sqrt(3) * arr[..., 7] - arr[..., 9]) ** 2)
        elif key == "gauge4":
            value = np.sqrt((arr[..., 10] - np.sqrt(6) * arr[..., 12] + arr[..., 14]) ** 2 + 2 * (arr[..., 11] - arr[..., 13]) ** 2)
        elif key == "res1d":
            rotated = rdht(arr, N, D, "fwd", "1")
            value = np.sqrt(energy(rotated, N, D, "ac")) - np.sqrt(energy(rotated, N, D, "1d"))
        elif key == "lowpass":
            value = arr[..., 0]
        else:
            raise ValueError(f"unknown invariant {name!r}")
        values.append(value)
    return np.stack(values, axis=-1), names
