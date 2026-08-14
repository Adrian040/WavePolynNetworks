"""Direct translation of ``idhtqt.m``."""

from typing import Sequence

import numpy as np

from ._quadtree import BitReader, dht_matrices, rotate_block, validate_image


def idhtqt(Z, qtbits: Sequence[bool], M: int, flag: str = "n") -> np.ndarray:
    coded = validate_image(Z, int(M))
    filters, windows = dht_matrices(int(M))
    output = np.zeros_like(coded)
    weight = np.zeros_like(coded)
    reader = BitReader(list(np.asarray(qtbits, dtype=bool).ravel()))
    rotated_mode = flag.lower().startswith("r")

    def visit(row_block: int, col_block: int, level: int) -> None:
        split = reader.read() if level > 1 else False
        if split:
            visit(2 * row_block, 2 * col_block, level - 1)
            visit(2 * row_block + 1, 2 * col_block, level - 1)
            visit(2 * row_block, 2 * col_block + 1, level - 1)
            visit(2 * row_block + 1, 2 * col_block + 1, level - 1)
            return
        block = 2 ** (level + 1)
        border = block - 4
        filt, window = filters[level - 1], windows[level - 1]
        top, left = row_block * block, col_block * block
        bottom, right = top + block, left + block
        x_top, x_left = max(0, top - border), max(0, left - border)
        x_bottom, x_right = min(coded.shape[0], bottom + border), min(coded.shape[1], right + border)
        h_top, h_left = x_top + border - top, x_left + border - left
        h_bottom = filt.shape[0] + x_bottom - border - bottom
        h_right = filt.shape[0] + x_right - border - right
        coeff = coded[top:bottom, left:right]
        if rotated_mode:
            coeff = rotate_block(coeff, inverse=True)
        reconstructed = filt[h_top:h_bottom] @ (coeff if rotated_mode else coeff / 255.0) @ filt[h_left:h_right].T
        output[x_top:x_bottom, x_left:x_right] += reconstructed
        weight[x_top:x_bottom, x_left:x_right] += window[h_top:h_bottom, h_left:h_right]

    block = 2 ** (M + 1)
    for row in range(coded.shape[0] // block):
        for col in range(coded.shape[1] // block):
            visit(row, col, int(M))
    if reader.position != len(reader.bits):
        raise ValueError("qtbits contains unused trailing decisions")
    return np.divide(output, weight, out=np.zeros_like(output), where=weight != 0)


__all__ = ["idhtqt"]
