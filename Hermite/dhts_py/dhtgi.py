"""Direct translation of ``dhtgi.m``."""

from typing import Iterable

import numpy as np

from .energy import energy
from .rdht import rdht


_DEFAULT_INVARIANTS = (
    "gauge1", "gauge2", "gauge3", "gauge4", "dflatness", "cornerness",
    "res1d", "umblicity", "laplacian", "flowlinec", "meancurv",
)


def dhtgi(Y, N, D=None, invr_name: str | Iterable[str] | None = None):
    if isinstance(Y, (list, tuple)):
        scale = np.atleast_1d(N)
        if scale.size > 1:
            base_n, degree = int(scale[0]), int(scale[1])
        else:
            base_n, degree = 8, int(scale[0])
        names_arg = D if D is not None else invr_name
        results = []
        for level, values in enumerate(Y):
            arr_level = np.asarray(values, dtype=float)
            stack = arr_level if level == len(Y) - 1 else np.concatenate((np.zeros(arr_level.shape[:2] + (1,)), arr_level), axis=2)
            results.append(dhtgi(stack, base_n if level == 0 else 6, degree, names_arg)[0])
        names = _DEFAULT_INVARIANTS if names_arg is None else ([names_arg] if isinstance(names_arg, str) else names_arg)
        return results, list(names)
    if D is None:
        raise ValueError("single-scale dhtgi requires D")
    N, D = int(N), int(D)
    names = list(_DEFAULT_INVARIANTS if invr_name is None else ([invr_name] if isinstance(invr_name, str) else invr_name))
    arr = np.asarray(Y, dtype=np.float64)
    values = []
    gradient = np.hypot(arr[..., 1], arr[..., 2]) if D >= 1 else None
    if D >= 2:
        trace = arr[..., 3] + arr[..., 5]
        hnorm = arr[..., 3] ** 2 + 2 * arr[..., 4] ** 2 + arr[..., 5] ** 2
    for name in names:
        key = name.lower()
        if key == "gradient":
            value = np.sqrt(N) * gradient
        elif key == "gauge1":
            value = gradient
        elif key == "laplacian":
            value = trace
        elif key == "dflatness":
            value = np.sqrt(hnorm)
        elif key in {"cornerness", "isophotec"}:
            directional = arr[..., 1] ** 2 * arr[..., 3] + 2 * arr[..., 1] * arr[..., 2] * arr[..., 4] + arr[..., 2] ** 2 * arr[..., 5]
            raw = directional - gradient * trace
            value = raw if key == "cornerness" else raw / (np.finfo(float).eps + gradient) ** 1.5
        elif key == "flowlinec":
            value = (arr[..., 4] * (arr[..., 1] ** 2 - arr[..., 2] ** 2) - arr[..., 1] * arr[..., 2] * (arr[..., 4] - arr[..., 3])) / (np.finfo(float).eps + gradient) ** 1.5
        elif key == "umblicity":
            value = (trace**2 - hnorm) / (np.finfo(float).eps + hnorm)
        elif key == "gaussianc":
            value = (trace**2 - hnorm) / (1 / 255 + gradient) ** 3
        elif key == "meancurv":
            value = 0.5 * trace / (1 / 255 + gradient) ** 1.5
        elif key in {"gauge2", "diaghess"}:
            value = np.sqrt((arr[..., 3] - arr[..., 5]) ** 2 + 2 * arr[..., 4] ** 2)
        elif key == "gauge3":
            value = np.sqrt((arr[..., 6] - np.sqrt(3) * arr[..., 8]) ** 2 + (np.sqrt(3) * arr[..., 7] - arr[..., 9]) ** 2)
        elif key == "gauge4":
            value = np.sqrt((arr[..., 10] - np.sqrt(6) * arr[..., 12] + arr[..., 14]) ** 2 + 2 * (arr[..., 11] - arr[..., 13]) ** 2)
        elif key == "res1d":
            rotated = rdht(arr, N, D, "fwd", "1")
            value = np.sqrt(energy(rotated, N, D, "ac")) - np.sqrt(energy(rotated, N, D, "1d"))
        elif key == "lowpass":
            value = arr[..., 0]
        else:
            raise ValueError(f"unknown invariant {name!r}")
        values.append(value)
    return np.stack(values, axis=-1), names


__all__ = ["dhtgi"]
