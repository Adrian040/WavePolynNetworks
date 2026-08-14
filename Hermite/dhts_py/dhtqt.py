"""Direct translation of ``dhtqt.m``."""

import numpy as np

from ._quadtree import dht_matrices, matlab_round, rotate_block, validate_image


def dhtqt(Z, M: int, flag: str = "n", k: float = 0.5):
    image = validate_image(Z, int(M))
    filters, _ = dht_matrices(int(M))
    output = np.zeros_like(image)
    bits = []
    rotated_mode = flag.lower().startswith("r")

    def visit(row_block: int, col_block: int, level: int) -> None:
        block = 2 ** (level + 1)
        border = block - 4
        filt = filters[level - 1]
        top, left = row_block * block, col_block * block
        bottom, right = top + block, left + block
        x_top, x_left = max(0, top - border), max(0, left - border)
        x_bottom, x_right = min(image.shape[0], bottom + border), min(image.shape[1], right + border)
        h_top, h_left = x_top + border - top, x_left + border - left
        h_bottom = filt.shape[0] + x_bottom - border - bottom
        h_right = filt.shape[0] + x_right - border - right
        hrow, hcol = filt[h_top:h_bottom], filt[h_left:h_right]
        patch = image[x_top:x_bottom, x_left:x_right]
        coeff = hrow.T @ patch @ hcol
        if rotated_mode:
            coeff = rotate_block(coeff)
        split = False
        if level > 1:
            a, b, c, d = 0.6, 0.15, 0.5 * level / 4, 0.1
            if rotated_mode:
                energy_main = np.sum(coeff[1:, 0] ** 2)
                error = np.sum(coeff[0, 2:] ** 2)
                dc = max(abs(coeff[0, 0]), np.finfo(float).eps)
                contrast = 0.1 * (b + abs(dc**a - c**a) ** (1 / a) / (dc**a + c**a) ** (1 / a))
                threshold = 0.01 * max(contrast, np.sqrt(energy_main) ** d * contrast ** (1 - d))
                split = error > threshold
            else:
                total = float(hrow[:, 0] @ (patch**2) @ hcol[:, 0])
                represented = float(np.sum(coeff**2))
                error = np.sqrt(max(total - represented, 0))
                dc = max(abs(coeff[0, 0]), np.finfo(float).eps)
                contrast = 0.2 * (b + abs(dc**a - c**a) ** (1 / a) / (dc**a + c**a) ** (1 / a))
                threshold = float(k) * max(contrast, np.sqrt(represented) ** d * contrast ** (1 - d))
                split = error > threshold
            bits.append(bool(split))
        if split:
            visit(2 * row_block, 2 * col_block, level - 1)
            visit(2 * row_block + 1, 2 * col_block, level - 1)
            visit(2 * row_block, 2 * col_block + 1, level - 1)
            visit(2 * row_block + 1, 2 * col_block + 1, level - 1)
        elif rotated_mode:
            coded = np.zeros_like(coeff)
            coded[:, 0] = coeff[:, 0]
            coded[0, 1] = coeff[0, 1]
            output[top:bottom, left:right] = coded
        else:
            output[top:bottom, left:right] = matlab_round(255 * coeff)

    block = 2 ** (M + 1)
    for row in range(image.shape[0] // block):
        for col in range(image.shape[1] // block):
            visit(row, col, int(M))
    return output, np.asarray(bits, dtype=bool)


__all__ = ["dhtqt"]
