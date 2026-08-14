"""Direct translation of ``matshow.m``."""

import numpy as np


def matshow(X, k: float = 3, *, ax=None, show: bool = False) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64).copy()
    if arr.ndim == 2:
        arr = arr[..., None]
    for channel in range(arr.shape[2]):
        std = np.std(arr[..., channel])
        arr[..., channel] = 0.5 if std == 0 else (arr[..., channel] - np.mean(arr[..., channel])) / (k * std) + 0.5
    result = np.clip(np.squeeze(arr), 0, 1)
    if ax is not None or show:
        import matplotlib.pyplot as plt

        target = ax or plt.subplots()[1]
        target.imshow(result, cmap="gray")
        target.axis("off")
        if show:
            plt.show()
    return result


__all__ = ["matshow"]
