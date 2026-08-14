"""Direct translation of ``im2qtb.m``."""

from typing import Sequence

import numpy as np

from ._quadtree import BitReader, validate_image


def im2qtb(Z, qtbits: Sequence[bool], M=4) -> list[np.ndarray]:
    image = np.asarray(Z)
    if isinstance(M, (list, tuple)):
        blocks = [np.asarray(value).copy() for value in M]
        levels = len(blocks)
    else:
        levels = int(M)
        blocks = [np.empty((2 ** (level + 2), 2 ** (level + 2), 0), dtype=image.dtype) for level in range(levels)]
    validate_image(image, levels)
    reader = BitReader(list(np.asarray(qtbits, dtype=bool).ravel()))

    def visit(row: int, col: int, level: int) -> None:
        block = 2 ** (level + 1)
        split = reader.read() if level > 1 else False
        if split:
            visit(2 * row, 2 * col, level - 1)
            visit(2 * row + 1, 2 * col, level - 1)
            visit(2 * row, 2 * col + 1, level - 1)
            visit(2 * row + 1, 2 * col + 1, level - 1)
        else:
            patch = image[row * block : (row + 1) * block, col * block : (col + 1) * block]
            blocks[level - 1] = np.concatenate((blocks[level - 1], patch[..., None]), axis=2)

    root = 2 ** (levels + 1)
    for row in range(image.shape[0] // root):
        for col in range(image.shape[1] // root):
            visit(row, col, levels)
    return blocks


__all__ = ["im2qtb"]
