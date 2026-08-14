"""Direct translation of ``dhtshow.m``."""

from typing import Sequence

import numpy as np

from .dhtord import dhtord
from .edht import edht


def dhtshow(Y, N: int | Sequence[int], D: int | None = None, *, ax=None, show: bool = False) -> np.ndarray:
    if isinstance(Y, (list, tuple)):
        if np.atleast_1d(N).size > 1:
            base_n, degree = map(int, np.atleast_1d(N)[:2])
        else:
            base_n, degree = 8, int(N)
        tiles = []
        for level, values in enumerate(Y):
            scale_n = base_n if level == 0 else 6
            arr = np.asarray(values)
            if level < len(Y) - 1:
                arr = np.concatenate((np.ones(arr.shape[:2] + (1,) + arr.shape[3:]), arr), axis=2)
            tiles.append(dhtshow(arr, scale_n, degree))
        height, width = sum(tile.shape[0] for tile in tiles), sum(tile.shape[1] for tile in tiles)
        channels = tiles[0].shape[2:] if tiles[0].ndim > 2 else ()
        mosaic = np.ones((height, width) + channels, dtype=float)
        row = col = 0
        for tile in tiles:
            mosaic[row : row + tile.shape[0], col : col + tile.shape[1]] = tile
            row += tile.shape[0]
            col += tile.shape[1]
    else:
        arr = np.asarray(Y, dtype=np.float64)
        degree = int(D if D is not None else N)
        if arr.ndim == 4:
            mosaic = np.stack([np.clip(dhtshow(arr[..., k], int(N), degree), 0, 1) for k in range(arr.shape[3])], axis=-1)
        else:
            if arr.ndim == 2:
                arr = arr[..., None]
            rows, cols = arr.shape[:2]
            side = min(int(N), degree) + 1
            mosaic = np.ones((side * rows, side * cols), dtype=float)
            for channel, (horizontal, vertical) in enumerate(dhtord(int(N), degree, 2)):
                if channel >= arr.shape[2]:
                    break
                bounds = np.array([-0.2, 0.2]) * (2 if channel == 0 else 1) + (0.5 if channel == 0 else 0)
                mosaic[vertical * rows : (vertical + 1) * rows, horizontal * cols : (horizontal + 1) * cols] = edht(arr[..., channel], bounds)
    if ax is not None or show:
        import matplotlib.pyplot as plt

        target = ax or plt.subplots()[1]
        target.imshow(mosaic, cmap="gray", vmin=0, vmax=1)
        target.axis("off")
        if show:
            plt.show()
    return mosaic


__all__ = ["dhtshow"]
