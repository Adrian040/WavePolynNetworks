"""Direct translation of ``mdht2.m``."""

import numpy as np

from ._core import scale_degree
from .dht2 import dht2


def mdht2(X, D, M: int, cod: str = "", *args, shape: str | None = None, return_aux: bool = False):
    if shape is None and len(cod) > 3:
        shape, cod = cod.lower(), (str(args[0]) if args else "")
        args = args[1:] if args else ()
    shape = shape or "full"
    N, degree = scale_degree(D, 8)
    T = int(round(np.sqrt(N / 2)))
    predict = "p" in cod
    keep = cod.startswith("0")
    post_code = cod.replace("p", "")
    if post_code.startswith("0"):
        post_code = post_code[1:]
    current = np.asarray(X, dtype=np.float64)
    result, auxiliary = [], []
    for level in range(int(M)):
        prefix = "h" if level == 0 else ("L" if level == M - 1 else "l")
        call_args = [value[level] if isinstance(value, (list, tuple)) and len(value) == M else value for value in args]
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
        from .pdht import pdht

        result = pdht(result, degree, "fwd", shape)
    return (result, auxiliary) if return_aux else result


__all__ = ["mdht2"]
