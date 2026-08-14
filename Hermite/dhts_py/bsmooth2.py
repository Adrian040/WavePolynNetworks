"""Translation of ``bsmooth2.m`` (whose MATLAB function name is a typo)."""

import numpy as np

from .idht2 import idht2
from .mdht2 import mdht2
from .mdhti2 import mdhti2


def bsmooth2(X, S, maxgrad=None) -> np.ndarray:
    scales = np.atleast_1d(S).astype(float)
    k = np.ceil(np.log2(scales) / 2 - 1).astype(int)
    tau = (4 - scales / (4.0**k)) / 3.5
    M = int(k.max()) + 1
    code = "0E" if maxgrad is not None else "0"
    pyramid = mdht2(X, [2, 12], M, "repeat", code, *(() if maxgrad is None else (maxgrad,)))
    stage = []
    for level in range(M):
        indices = np.flatnonzero(k == level)
        target = np.asarray(X).shape[:2] if level == 0 else np.asarray(pyramid[level - 1]).shape[:2]
        reconstructed = []
        for index in indices[::-1]:
            if tau[index] != 0:
                reconstructed.append(idht2(pyramid[level], target, 2 if level == 0 else 6, 12, 1 if level == 0 else 2, "repeat", "s", tau[index]))
            else:
                reconstructed.append(idht2(np.asarray(pyramid[level])[..., :1], target, 2 if level == 0 else 6, 0, 1 if level == 0 else 2, "repeat"))
        stage.append(np.stack(reconstructed, axis=-1) if reconstructed else np.empty(target + (0,)))
    if M == 1:
        return stage[0]
    upper = mdhti2(stage[1:], np.asarray(X).shape[:2], [2, 0], "nosum", "repeat")
    return np.concatenate((stage[0], upper[..., ::-1]), axis=-1)


__all__ = ["bsmooth2"]
