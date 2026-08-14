"""Direct translation of ``zcross.m``."""

import numpy as np

from .rdht import rdht


def _neighbors8(X: np.ndarray, connectivity: int) -> np.ndarray:
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    padded = np.pad(X, 1, mode="edge")
    return np.stack([padded[1 + dr : 1 + dr + X.shape[0], 1 + dc : 1 + dc + X.shape[1]] for dr, dc in offsets], axis=-1)


def zcross(Y, N: int, D: int, conn: int = 8, CMIN=None, SMAX=None, *, scale_model=None):
    if min(D, N) < 3:
        raise ValueError("zcross requires coefficients through third order")
    rotated = rdht(np.asarray(Y)[..., :10], N, 3, "fwd", "grad")
    center = rotated[..., 3]
    neighbors = _neighbors8(center, int(conn))
    crossing = np.any((neighbors * center[..., None] < 0) & (np.abs(center[..., None]) <= np.abs(neighbors)), axis=-1)
    mask = crossing & (rotated[..., 6] < 0)
    if CMIN is None and SMAX is None:
        return mask
    if scale_model is None:
        raise FileNotFoundError("scale/contrast estimation needs scest.mat; pass its matrix with scale_model=")
    model = np.asarray(scale_model, dtype=float)
    column = N // 2 - 2
    scale = N * (model[0, column] * (rotated[..., 1] / rotated[..., 6]) + model[1, column])
    scale *= scale > 0
    contrast = 2.5 * rotated[..., 1] * np.sqrt(1 + scale * (8 / N))
    if SMAX is not None:
        mask &= (scale > 0) & (scale <= SMAX)
    if CMIN is not None:
        mask &= contrast >= CMIN
    return mask, scale, contrast


__all__ = ["zcross"]
