"""Direct translation of ``overshoot.m``."""

import numpy as np

from .dht2 import dht2
from .idht2 import idht2
from .imdht2 import imdht2
from .mdht2 import mdht2


def overshoot(X, M: int) -> np.ndarray:
    image = np.asarray(X, dtype=np.float64)
    Y0 = dht2(image, 2, 4, 1)
    alpha = 4.0
    tau = alpha / (alpha - 1) - 1 / np.log(alpha)
    padded = np.zeros(tuple(value + 2 for value in image.shape))
    padded[1:-1, 1:-1] = idht2(Y0, image.shape, 2, 4, 1, "s", tau)
    high = mdht2(np.log(4) * padded, [6, 1], M)
    low = mdht2(Y0[..., 0], [6, 0], M)
    high[-1][..., 0] = low[-1][..., 0]
    Y0[..., 0] = imdht2(high, Y0.shape[:2], [6, 1])
    return idht2(Y0, image.shape, 2, 4, 1)


__all__ = ["overshoot"]
