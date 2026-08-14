"""Direct translation of ``dht.m``."""

import numpy as np

from ._core import BOUNDARY_MODES, conv_axis, default_dim, integer, pad_axis
from .dhtmtx import dhtmtx


def dht(
    X: np.ndarray,
    N: int,
    D: int,
    T: int,
    dim: int | None = None,
    shape: str = "full",
    cod: str = "h",
    *args,
    return_lowpass: bool = False,
):
    """Compute the 1-D DHT along the MATLAB-style one-based ``dim``."""

    X = np.asarray(X, dtype=np.float64)
    N = integer("N", N)
    D = min(integer("D", D), N)
    T = integer("T", T, 1)
    if N and T > N:
        raise ValueError("T must satisfy 1 <= T <= N")
    if dim is None:
        dim = default_dim(X.shape)
    axis = int(dim) - 1
    if not 0 <= axis < X.ndim:
        raise ValueError("dim is outside the input dimensions")
    shape = shape.lower()
    if shape not in BOUNDARY_MODES:
        raise ValueError(f"invalid shape {shape!r}")

    work = X
    conv_mode = shape
    if shape in {"symm", "repeat", "asymm", "cyclic"}:
        work = pad_axis(work, N, axis, shape)
        conv_mode = "valid"

    H = dhtmtx(N, D)
    code = cod or " "
    if code[0].lower() == "l":
        H = H / np.sqrt(0.75 ** np.arange(H.shape[1]))[None, :]
        code = code[1:] or " "
    elif code[0].lower() == "h":
        code = code[1:] or " "

    channels = []
    for n in range(D + 1):
        y = conv_axis(work, H[:, n], axis, conv_mode)
        channels.append(y.take(range(0, y.shape[axis], T), axis=axis))
    Y = np.stack(channels, axis=-1)

    if code and code[0].lower() == "s":
        if not args:
            raise ValueError("cod='s' requires tau")
        tau = np.asarray(args[0], dtype=float)
        for n in range(1, D + 1):
            Y[..., n] *= tau**n

    if return_lowpass:
        Y0 = Y[..., 0]
        Y = Y if "0" in code else Y[..., 1:]
        return Y, Y0
    return Y


__all__ = ["dht"]
