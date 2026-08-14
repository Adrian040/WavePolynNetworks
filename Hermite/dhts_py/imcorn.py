"""Direct translation of ``imcorn.m``."""

import numpy as np

from .dht2 import dht2


def imcorn(X, N: int = 2, kappa: float = 0.1) -> np.ndarray:
    # The MATLAB source unconditionally sets N=2; preserve that behavior.
    Y = dht2(np.asarray(X, dtype=np.float64), 2, 1, 1, "repeat")
    products = np.stack((Y[..., 1] * Y[..., 2], Y[..., 1] ** 2, Y[..., 2] ** 2), axis=-1)
    smooth = np.stack([dht2(products[..., k], 2, 0, 1, "repeat")[..., 0] for k in range(3)], axis=-1)
    return smooth[..., 1] * smooth[..., 2] - smooth[..., 0] ** 2 - kappa * (smooth[..., 1] + smooth[..., 2]) ** 2


__all__ = ["imcorn"]
