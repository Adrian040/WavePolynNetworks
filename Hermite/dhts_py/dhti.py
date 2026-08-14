"""Direct translation of ``dhti.m``."""

from typing import Sequence

import numpy as np

from .idht import idht


def dhti(
    Y: np.ndarray,
    xsiz: int | Sequence[int],
    N: int,
    D: int,
    T: int,
    dim: int | None = None,
    sumopt: str = "nosum",
    cod: str = " ",
    *args,
    shape: str | None = None,
) -> np.ndarray:
    """Interpolate individual 1-D coefficient contributions."""

    arr = np.asarray(Y, dtype=np.float64)
    ncoeff = arr.shape[-1] if arr.ndim > 1 else 1
    D1 = min(D, N)
    extra = max(0, ncoeff - (D1 + 1))
    pieces = []
    for channel in range(ncoeff):
        order = 0 if channel <= extra else channel - extra
        one = np.zeros(arr.shape[:-1] + (D1 + 1,), dtype=np.float64)
        one[..., order] = arr[..., channel]
        pieces.append(idht(one, xsiz, N, D1, T, dim, cod, *args, shape=shape))

    option = sumopt.lower()
    if option == "nosum":
        return np.stack(pieces, axis=-1)
    if option == "high":
        low_count = extra + 1
        low = np.sum(pieces[:low_count], axis=0)
        high = np.sum(pieces[low_count:], axis=0) if low_count < len(pieces) else np.zeros_like(low)
        return np.stack((low, high), axis=-1)
    if option == "predict":
        raise NotImplementedError("SUMOPT='predict' was not implemented in dhti.m")
    raise ValueError(f"unrecognized sumopt {sumopt!r}")


__all__ = ["dhti"]
