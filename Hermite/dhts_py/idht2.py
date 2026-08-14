"""Direct translation of ``idht2.m``."""

from math import ceil, comb
from typing import Sequence

import numpy as np
from scipy.signal import convolve2d

from ._core import expand_coefficients_2d, integer, parse_dht2_options, synthesis_filters
from .dhtmtx import dhtmtx
from .dhtord import dhtord


def idht2(
    Y: np.ndarray,
    xsiz: Sequence[int],
    N: int,
    D: int,
    T: int,
    cod: str = " ",
    *args,
    shape: str | None = None,
    return_aux: bool = False,
):
    """Synthesize an image from 2-D DHT coefficients."""

    parsed_shape, code, args = parse_dht2_options(cod, args)
    shape = (shape or parsed_shape).lower()
    target = tuple(int(v) for v in xsiz[:2])
    arr = np.asarray(Y, dtype=np.float64)

    if len(xsiz) == 3:
        bands = [
            idht2(arr[..., k], target, N, D, T, shape, code, *args, return_aux=return_aux)
            for k in range(int(xsiz[2]))
        ]
        if return_aux:
            return np.stack([value[0] for value in bands], axis=-1), [value[1] for value in bands]
        return np.stack(bands, axis=-1)
    if arr.ndim == 2:
        arr = arr[..., None]

    quincunx = "q" in code.lower()
    low_resolution = code[0].lower() == "l"
    if code[0].lower() in {"l", "h"}:
        code = code[1:] or " "
    D = min(integer("D", D), 2 * integer("N", N))
    T = integer("T", T, 1)

    if quincunx:
        T2 = 2 * T
        H, G = dhtmtx(N, min(D, N), T2)
        H = H[::-1, :]
        start = ceil((N + 1) / 2) - T * ceil(N / T2) + ceil(N / 2)
        stop = ceil((N + 1) / 2) + T2 - T * ceil(N / T2) + ceil(N / 2) - 1
        indices = np.mod(np.arange(start, stop + 1) - 1, N).astype(int)
        W = H[indices, 0] / G[indices, 0]
        W = np.outer(W, W)
        W = W + np.fft.fftshift(W)
        repetitions = (ceil(target[0] / T2), ceil(target[1] / T2))
        quincunx_weight = np.tile(W, repetitions)[: target[0], : target[1]]
        G = H
    else:
        G = synthesis_filters(N, min(D, N), T, low_resolution)

    aux = None
    key = code[0] if code else " "
    if key == "q":
        from .qdht import qdht

        arr = qdht(arr, "inv", code[1:], N, D, *args)
    elif key == "r":
        from .rdht import rdht

        arr, aux = rdht(arr, N, D, "inv", *(args or ("grad",)), return_theta=True)
    elif key == "d":
        from .ddht import ddht

        arr, aux = ddht(arr, N, D, "inv", *(args or ("grad",)), return_theta=True)
    elif key in {"s", "c"}:
        from .sdht2 import sdht2

        tau = args[0] if args else None
        theta = args[1] if len(args) > 1 else (None if key == "c" else ...)
        if theta is ...:
            arr, aux, _ = sdht2(arr, N, D, tau)
        else:
            arr, aux, _ = sdht2(arr, N, D, tau, theta)
    elif key == "g":
        offset = ceil(N / T)
        inner = arr[offset:-offset, offset:-offset, 0] if offset else arr[..., 0]
        mean, std = float(np.mean(inner)), float(np.std(inner))
        arr[..., 0] = 0.5 if std == 0 else (1 + np.tanh((arr[..., 0] - mean) / std)) / 2
    elif key == "e":
        from .edht import edht

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
        from .gauge import gauge

        sign = 1 if key == "E" else -1
        count = min(D, N)
        components = []
        for n in range(1, count + 1):
            a, b = gauge(arr, N, D, n, components=True)
            components.append(np.hypot(a, b))
        polynomial = sign * np.poly(np.ones(count))
        for n in range(1, count + 1):
            arr[..., 0] += polynomial[n] * components[n - 1]
    elif key == "p":
        if not args:
            raise ValueError("cod='p' requires the order pair [j,k]")
        horizontal, vertical = map(int, np.asarray(args[0]).ravel()[:2])
        orders = dhtord(N, D, 2)
        selected = np.flatnonzero(
            (orders[:, 0] >= horizontal) & (orders[:, 1] >= vertical)
        )
        if selected.size == 0:
            raise ValueError("higher-order coefficients are required for prediction")
        arr = arr[..., selected].copy()
        D = D - horizontal - vertical
        position = 0
        for total in range(min(D, 2 * N) + 1):
            for vertical_order in range(max(0, total - N), min(N, total) + 1):
                horizontal_order = total - vertical_order
                Cj = comb(horizontal_order + horizontal, horizontal)
                Ck = comb(vertical_order + vertical, vertical)
                arr[..., position] *= np.sqrt(
                    Cj * Ck * 0.75**total * 0.25 ** (horizontal + vertical)
                )
                position += 1

    expected = len(dhtord(N, D, 2))
    if arr.shape[2] < expected:
        raise ValueError(f"expected {expected} coefficient channels, got {arr.shape[2]}")

    arr, ysiz, t0, conv_mode = expand_coefficients_2d(arr, target, N, T, shape)
    rows = np.arange(t0, ysiz[0], T)
    cols = np.arange(t0, ysiz[1], T)
    if arr.shape[:2] != (len(rows), len(cols)):
        raise ValueError("coefficient dimensions do not match N, T, shape, and xsiz")

    X = np.zeros(target, dtype=np.float64)
    for channel, (horizontal, vertical) in enumerate(dhtord(N, D, 2)):
        yi = np.zeros(ysiz, dtype=np.float64)
        yi[np.ix_(rows, cols)] = arr[..., channel]
        X += convolve2d(yi, np.outer(G[:, vertical], G[:, horizontal]), mode=conv_mode)
    if quincunx:
        X = np.divide(X, quincunx_weight, out=np.zeros_like(X), where=quincunx_weight != 0)
    return (X, aux) if return_aux else X


__all__ = ["idht2"]
