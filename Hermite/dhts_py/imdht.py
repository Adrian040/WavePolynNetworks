"""Direct translation of ``imdht.m``."""

from typing import Sequence

import numpy as np

from ._core import scale_degree
from .idht import idht


def imdht(Y: Sequence[np.ndarray], xsiz: Sequence[int] | int, D, dim: int | None = None, cod: str = " ", *args, shape: str | None = None):
    if shape is None and len(cod) > 3:
        shape, cod = cod.lower(), (str(args[0]) if args else " ")
        args = args[1:] if args else ()
    shape = shape or "full"
    N0, degree = scale_degree(D, 8)
    T0 = int(round(np.sqrt(N0 / 2)))
    target = tuple(np.atleast_1d(xsiz).astype(int))
    if dim is None:
        dim = next((i + 1 for i, value in enumerate(target) if value > 1), 1)
    current = np.asarray(Y[-1], dtype=np.float64)
    for level in range(len(Y) - 2, -1, -1):
        saved = np.asarray(Y[level], dtype=np.float64)
        stage_shape = list(target)
        stage_shape[dim - 1] = saved.shape[dim - 1]
        low = idht(current, stage_shape, 6, degree, 2, dim, "l" + cod, *args, shape=shape)
        current = low if saved.size == 0 else np.concatenate((low[..., None], saved), axis=-1)
    return idht(current, target, N0, degree, T0, dim, "h" + cod, *args, shape=shape)


__all__ = ["imdht"]
