"""Direct translation of ``dht2.m``."""

from math import ceil

import numpy as np
from scipy.signal import convolve2d

from ._core import integer, pad_spatial, parse_dht2_options
from .dhtmtx import dhtmtx
from .dhtord import dhtord


def dht2(
    X: np.ndarray,
    N: int,
    D: int,
    T: int,
    cod: str = " ",
    *args,
    shape: str | None = None,
    return_aux: bool = False,
):
    """Compute the 2-D DHT in the channel order defined by ``dhtord.m``."""

    parsed_shape, code, args = parse_dht2_options(cod, args)
    shape = (shape or parsed_shape).lower()
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 3 and X.shape[2] > 1:
        bands = [
            dht2(X[..., k], N, D, T, shape, code, *args, return_aux=return_aux)
            for k in range(X.shape[2])
        ]
        if return_aux:
            return np.stack([value[0] for value in bands], axis=-1), [value[1] for value in bands]
        return np.stack(bands, axis=-1)
    if X.ndim != 2:
        raise ValueError("dht2 expects a 2-D image or a 3-D multiband image")

    N = integer("N", N)
    D = min(integer("D", D), 2 * N)
    T = integer("T", T, 1)
    H = dhtmtx(N, D)
    work = X
    conv_mode = shape
    if shape in {"symm", "repeat", "asymm", "cyclic"}:
        work = pad_spatial(X, N, shape, 2)
        conv_mode = "valid"
    elif shape not in {"full", "same", "valid"}:
        raise ValueError(f"invalid shape {shape!r}")

    msflag = 0
    if code[0].lower() == "l":
        H = H / np.sqrt(0.75 ** np.arange(H.shape[1]))[None, :]
        msflag = 1 + int(code[0] == "L")
        code = code[1:] or " "
    elif code[0].lower() == "h":
        code = code[1:] or " "

    channels = []
    for horizontal, vertical in dhtord(N, D, 2):
        kernel = np.outer(H[:, vertical], H[:, horizontal])
        channels.append(convolve2d(work, kernel, mode=conv_mode)[::T, ::T])
    Y = np.stack(channels, axis=-1)

    if not code.strip():
        return (Y, None) if return_aux else Y

    key = code[0]
    aux = None
    if key == "q":
        from .qdht import qdht

        Y = qdht(Y, "fwd", code[1:], N, D, *args)
    elif key == "r":
        from .rdht import rdht

        Y, aux = rdht(Y, N, D, "fwd", *(args or ("grad",)), return_theta=True)
    elif key == "d":
        from .ddht import ddht

        Y, aux = ddht(Y, N, D, "fwd", *(args or ("grad",)), return_theta=True)
    elif key == "s":
        from .sdht2 import sdht2

        Y, aux, _ = sdht2(Y, N, D, *args)
    elif key == "c":
        from .sdht2 import sdht2

        Y, aux, _ = sdht2(Y, N, D, None, None)
    elif key == "m":
        from .dhtmorph import dhtmorph

        Y, aux = dhtmorph(Y, N, D, *args)
    elif key == "G":
        offset = ceil(N / T)
        inner = Y[offset:-offset, offset:-offset, 0] if offset else Y[..., 0]
        mean, std = float(np.mean(inner)), float(np.std(inner))
        Y[..., 0] = 0.5 if std == 0 else (1 + np.tanh((Y[..., 0] - mean) / std)) / 2
    elif key == "i":
        from .dhtgi import dhtgi

        values, aux = dhtgi(Y, N, D, *(args or ()))
        Y = np.concatenate((Y[..., :1], values), axis=-1)
    elif key == "g":
        from .gauge import gauge

        result = np.zeros(Y.shape[:2] + (D + 1,), dtype=np.float64)
        result[..., 0] = Y[..., 0]
        for n in range(1, min(D, N) + 1):
            a, b = gauge(Y, N, D, n, components=True)
            result[..., n] = np.hypot(a, b)
        Y = result
    elif key == "e":
        from .edht import edht

        Y[..., 0] = edht(Y[..., 0], *(args or ()))
    elif key == "x":
        from .xdht2 import xdht2

        Y, aux = xdht2(Y, N, D, *args)
    elif key == "E":
        from .sdht2 import sdht2

        if D < 1:
            raise ValueError("erosion requires D > 0")
        gradient = np.hypot(Y[..., 1], Y[..., 2])
        mask = gradient > args[0] if args else np.ones(Y.shape[:2], bool)
        if args and msflag < 2:
            Y, _, _ = sdht2(Y, N, D, (~mask).astype(float))
        Y[..., 0][mask] -= gradient[mask] / np.sqrt(2.0)
        aux = mask
    return (Y, aux) if return_aux else Y


__all__ = ["dht2"]
