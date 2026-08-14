"""Direct translation of ``imdht2.m``."""

from typing import Sequence

import numpy as np

from ._core import scale_degree
from .idht2 import idht2


def imdht2(Y: Sequence[np.ndarray], xsiz: Sequence[int], D, cod: str = "", *args, shape: str | None = None):
    if shape is None and len(cod) > 3:
        shape, cod = cod.lower(), (str(args[0]) if args else "")
        args = args[1:] if args else ()
    shape = shape or "full"
    N0, degree = scale_degree(D, 8)
    T0 = int(round(np.sqrt(N0 / 2)))
    keep = cod.startswith("0")
    post_code = cod[1:] if keep else cod
    levels = list(Y)
    if post_code.startswith("p"):
        from .pdht import pdht

        levels = pdht(levels, degree, "inv", shape, post_code[1:], *args)
        post_code = "q" if "q" in post_code else ""
    current = np.asarray(levels[-1], dtype=np.float64)
    for level in range(len(levels) - 2, -1, -1):
        saved = np.asarray(levels[level], dtype=np.float64)
        stage_shape = saved.shape[:2]
        call_args = [value[level + 1] if isinstance(value, (list, tuple)) and len(value) == len(levels) else value for value in args]
        low = idht2(current, stage_shape, 6, degree, 2, shape, "l" + post_code, *call_args)
        if low.ndim == 2:
            low = low[..., None]
        current = np.concatenate((low, saved), axis=2)
    call_args = [value[0] if isinstance(value, (list, tuple)) and len(value) == len(levels) else value for value in args]
    return idht2(current, xsiz, N0, degree, T0, shape, "h" + post_code, *call_args)


__all__ = ["imdht2"]
