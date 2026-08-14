"""Direct translation of ``qtb2im.m``."""

from typing import Sequence

import numpy as np

from ._quadtree import BitReader


def qtb2im(blocks: Sequence[np.ndarray], qtbits: Sequence[bool], xsiz: Sequence[int], M: int | None = None) -> np.ndarray:
    levels = len(blocks) if M is None else int(M)
    queues = [list(np.moveaxis(np.asarray(value), 2, 0)) for value in blocks]
    output = np.zeros(tuple(int(value) for value in xsiz[:2]), dtype=np.result_type(*[np.asarray(value).dtype for value in blocks]))
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
            if not queues[level - 1]:
                raise ValueError(f"not enough blocks at level {level}")
            output[row * block : (row + 1) * block, col * block : (col + 1) * block] = queues[level - 1].pop(0)

    root = 2 ** (levels + 1)
    for row in range(output.shape[0] // root):
        for col in range(output.shape[1] // root):
            visit(row, col, levels)
    return output


__all__ = ["qtb2im"]
