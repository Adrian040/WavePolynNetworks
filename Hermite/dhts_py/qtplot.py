"""Direct translation of ``qtplot.m``."""

from typing import Sequence

import numpy as np

from ._quadtree import BitReader, validate_image


def qtplot(X, qtbits: Sequence[bool], M: int = 4, circle: bool = False, *plot_args, ax=None):
    image = validate_image(X, int(M))
    reader = BitReader(list(np.asarray(qtbits, dtype=bool).ravel()))
    xs, ys, centers_x, centers_y = [np.nan], [np.nan], [np.nan], [np.nan]

    def visit(row: int, col: int, level: int) -> None:
        split = reader.read() if level > 1 else False
        block = 2 ** (level + 1)
        if split:
            if not circle:
                xedge, yedge = [col * block, (col + 1) * block], [row * block, (row + 1) * block]
                xmid, ymid = (col + 0.5) * block, (row + 0.5) * block
                xs.extend([*xedge, np.nan, xmid, xmid, np.nan])
                ys.extend([ymid, ymid, np.nan, *yedge, np.nan])
            visit(2 * row, 2 * col, level - 1)
            visit(2 * row + 1, 2 * col, level - 1)
            visit(2 * row, 2 * col + 1, level - 1)
            visit(2 * row + 1, 2 * col + 1, level - 1)
        elif circle:
            cx, cy = (col + 0.5) * block, (row + 0.5) * block
            radius = np.sqrt(2 * (4**level - 1))
            theta = np.arange(0, 2 * np.pi + np.pi / block, np.pi / block)
            xs.extend([*(cx + radius * np.cos(theta)), np.nan])
            ys.extend([*(cy + radius * np.sin(theta)), np.nan])
            centers_x.extend([cx, np.nan])
            centers_y.extend([cy, np.nan])

    root = 2 ** (M + 1)
    for row in range(image.shape[0] // root):
        for col in range(image.shape[1] // root):
            visit(row, col, int(M))
    if ax is not None:
        ax.imshow(image, cmap="gray")
        ax.plot(xs, ys, *(plot_args or ("b-",)))
        if circle:
            ax.plot(centers_x, centers_y, "y.")
    return np.asarray(xs), np.asarray(ys)


__all__ = ["qtplot"]
