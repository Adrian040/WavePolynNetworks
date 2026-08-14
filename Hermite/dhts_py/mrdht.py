"""Direct translation of ``mrdht.m``."""

from typing import Sequence

import numpy as np

from ._core import scale_degree
from .rdht import rdht


def mrdht(Y: Sequence[np.ndarray], D, tdir: str = "fwd", theta="grad", *, return_theta: bool = False):
    N, degree = scale_degree(D, 8)
    angles = [theta] * len(Y) if isinstance(theta, str) or np.isscalar(theta) else list(theta)
    result, used = [], []
    for level, saved in enumerate(Y):
        if level:
            N = 6
        arr = np.asarray(saved, dtype=np.float64)
        work = arr if level == len(Y) - 1 else np.concatenate((np.ones(arr.shape[:2] + (1,)), arr), axis=2)
        transformed, angle = rdht(work, N, degree, tdir, angles[level], return_theta=True)
        result.append(transformed if level == len(Y) - 1 else transformed[..., 1:])
        used.append(angle)
    return (result, used) if return_theta else result


__all__ = ["mrdht"]
