"""Direct translation of ``mrdht2.m``."""

from typing import Sequence

import numpy as np

from ._core import scale_degree
from .rdht import rdht


def mrdht2(Y: Sequence[np.ndarray], D, *args) -> list[np.ndarray]:
    N, degree = scale_degree(D, 2)
    result = []
    for level, saved in enumerate(Y):
        if level == 1:
            N = 6
        arr = np.asarray(saved, dtype=np.float64)
        work = arr if level == len(Y) - 1 else np.concatenate((np.zeros(arr.shape[:2] + (1,)), arr), axis=2)
        transformed = rdht(work, N, degree, *args)
        result.append(transformed if level == len(Y) - 1 else transformed[..., 1:])
    return result


__all__ = ["mrdht2"]
