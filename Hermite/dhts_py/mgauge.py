"""Direct translation of ``mgauge.m``."""

from typing import Sequence

import numpy as np

from ._core import scale_degree
from .gauge import gauge


def mgauge(Y: Sequence[np.ndarray], D, L=1, *, components: bool = False):
    N, degree = scale_degree(D, 8)
    first, second = [], []
    for level, saved in enumerate(Y):
        if level:
            N = 6
        arr = np.asarray(saved, dtype=np.float64)
        work = arr if level == len(Y) - 1 else np.concatenate((np.zeros(arr.shape[:2] + (1,)), arr), axis=2)
        if components:
            a, b = gauge(work, N, degree, L, components=True)
            first.append(a)
            second.append(b)
        else:
            first.append(gauge(work, N, degree, L))
    return (first, second) if components else first


__all__ = ["mgauge"]
