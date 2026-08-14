"""Direct translation of ``bsmooth.m``."""

from math import floor, log2

import numpy as np

from .dht import dht
from .idht import idht
from .imdht import imdht
from .mdht import mdht


def bsmooth(X, N: int) -> np.ndarray:
    M = floor(log2(np.sqrt(N))) - 1
    if M <= 0:
        return np.squeeze(dht(X, N, 0, 1, shape="same"), axis=-1)
    tau = (4 ** (M + 1) - N) / (3 * 4**M)
    pyramid = mdht(X, 0, M)
    last = np.asarray(pyramid[-1])
    coeff = dht(np.squeeze(last[..., 0] if last.shape[-1] == 1 else last), 6, 6, 2)
    coeff *= tau ** np.arange(7)
    pyramid[-1] = idht(coeff, np.squeeze(last).shape, 6, 6, 2)
    return imdht(pyramid, np.asarray(X).shape, 0)


__all__ = ["bsmooth"]
