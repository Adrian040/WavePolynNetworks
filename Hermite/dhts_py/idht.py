"""Direct translation of ``idht.m``."""

from typing import Sequence

import numpy as np
from scipy.signal import convolve2d

from ._core import (
    default_dim,
    expand_1d_coefficients,
    integer,
    shape_tuple,
    synthesis_filters,
    synthesis_geometry,
)


def idht(
    Y: np.ndarray,
    xsiz: int | Sequence[int],
    N: int,
    D: int,
    T: int,
    dim: int | None = None,
    cod: str = " ",
    *args,
    shape: str | None = None,
) -> np.ndarray:
    """Synthesize a signal from 1-D DHT coefficients."""

    target_shape = shape_tuple(xsiz)
    if dim is None:
        dim = default_dim(target_shape)
    axis = int(dim) - 1
    if not 0 <= axis < len(target_shape):
        raise ValueError("dim is outside xsiz")

    if shape is None and isinstance(cod, str) and len(cod) > 3:
        shape, cod = cod.lower(), (str(args[0]) if args else " ")
        args = args[1:] if args else ()
    shape = (shape or "full").lower()
    code = cod or " "
    low_resolution = code[0].lower() == "l"
    if code[0].lower() in {"l", "h"}:
        code = code[1:] or " "
    D = min(integer("D", D), integer("N", N))
    T = integer("T", T, 1)
    G = synthesis_filters(N, D, T, low_resolution)

    arr = np.asarray(Y, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if len(target_shape) > 1:
        arr = np.moveaxis(arr, axis, 0)
    if arr.shape[-1] < D + 1:
        raise ValueError(f"expected at least {D + 1} coefficient channels")

    if code[0].lower() == "s":
        if not args:
            raise ValueError("cod='s' requires tau")
        tau = np.asarray(args[0], dtype=float)
        arr = arr.copy()
        for n in range(1, D + 1):
            arr[..., n] *= tau**n

    arr, t0 = expand_1d_coefficients(arr, target_shape[axis], N, T, shape)
    ylen, conv_mode = synthesis_geometry(target_shape[axis], N, shape)
    flat = arr.reshape(arr.shape[0], -1, arr.shape[-1])
    X = np.zeros((target_shape[axis], flat.shape[1]), dtype=np.float64)
    lattice = np.arange(t0, ylen, T)
    if len(lattice) != flat.shape[0]:
        raise ValueError("coefficient dimensions do not match N, T, shape, and xsiz")

    for n in range(D + 1):
        yi = np.zeros((ylen, flat.shape[1]), dtype=np.float64)
        yi[lattice, :] = flat[:, :, n]
        X += convolve2d(yi, G[:, n, None], mode=conv_mode)

    moved_shape = (target_shape[axis],) + tuple(
        target_shape[i] for i in range(len(target_shape)) if i != axis
    )
    X = X.reshape(moved_shape)
    return np.moveaxis(X, 0, axis) if len(target_shape) > 1 else X.ravel()


__all__ = ["idht"]
