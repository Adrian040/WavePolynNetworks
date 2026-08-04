"""Quadtree DHT coding, reconstruction, block packing, and plotting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import convolve2d

from ._core import dhtord
from ._steering import rdht


Array = np.ndarray


def _matlab_round(x: Array) -> Array:
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _dht_matrices(M: int) -> tuple[list[Array], list[Array]]:
    if M < 1:
        raise ValueError("M must be >= 1")
    h = np.array(
        [[1, 1, 1, 1], [3, 1, -1, -3], [3, -1, -1, 3], [1, -1, 1, -1]],
        dtype=float,
    )
    first = h[:, :4] * (np.sqrt(h[:4, 0]) / 2.0**3)[None, :]
    filters = [first]
    windows = [np.outer(first[:, 0], first[:, 0])]
    for level in range(2, M + 1):
        N = 4**level - 1
        T = 2 ** (level + 1)
        border = T - 4
        start = int((N - T + 1) / 2 - border)
        stop = int((N + T + 1) / 2 + border)
        smooth = np.convolve(h[:, 0], [1, 1])
        detail = np.convolve(h[:, -1], [1, -1])
        old = h
        h = np.column_stack(
            [
                *[np.convolve(smooth, old[:, j]) for j in range(old.shape[1])],
                *[np.convolve(detail, old[:, j]) for j in range(old.shape[1])],
            ]
        )
        smooth = np.convolve(h[:, 0], [1, 1])
        h = np.column_stack([np.convolve(smooth, h[:, j]) for j in range(h.shape[1])])
        selected = h[start:stop, :T] * (np.sqrt(h[:T, 0]) / 2.0**N)[None, :]
        filters.append(selected)
        windows.append(np.outer(selected[:, 0], selected[:, 0]))
    return filters, windows


def _matrix_to_stack(matrix: Array) -> Array:
    T = matrix.shape[0]
    orders = dhtord(T - 1, 2 * (T - 1), 2)
    return np.asarray([matrix[v, h] for h, v in orders], dtype=float)[None, None, :]


def _stack_to_matrix(stack: Array, T: int) -> Array:
    matrix = np.zeros((T, T), dtype=float)
    for value, (horizontal, vertical) in zip(np.asarray(stack).ravel(), dhtord(T - 1, 2 * (T - 1), 2)):
        matrix[vertical, horizontal] = value
    return matrix


def _rotate_block(matrix: Array, inverse: bool = False) -> Array:
    T = matrix.shape[0]
    stack = _matrix_to_stack(matrix)
    if inverse:
        angle = 2 * np.pi * matrix[0, 1]
        stack[..., list(np.flatnonzero((dhtord(T - 1, 2 * (T - 1), 2) == (1, 0)).all(axis=1)))] = 0
        rotated = rdht(stack, T - 1, 2 * (T - 1), "inv", -angle)
    else:
        angle = np.arctan2(matrix[0, 1], matrix[1, 0])
        # rotcoef.m stores matrix rows as y-order and columns as x-order;
        # this is RDHT's normalized recurrence evaluated at the negative angle.
        rotated = rdht(stack, T - 1, 2 * (T - 1), "fwd", -angle)
    result = _stack_to_matrix(rotated, T)
    if not inverse:
        result[0, 1] = angle / (2 * np.pi)
    return result


@dataclass
class _BitReader:
    bits: list[bool]
    position: int = 0

    def read(self) -> bool:
        if self.position >= len(self.bits):
            raise ValueError("qtbits ended before the quadtree traversal completed")
        value = bool(self.bits[self.position])
        self.position += 1
        return value


def _validate_image(image: Array, M: int) -> Array:
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError("quadtree functions require a 2-D array")
    block = 2 ** (M + 1)
    if arr.shape[0] % block or arr.shape[1] % block:
        raise ValueError(f"image dimensions must be multiples of {block} for M={M}")
    return arr


def dhtqt(Z: Array, M: int, flag: str = "n", k: float = 0.5):
    """DHT-quadtree decomposition."""

    image = _validate_image(Z, int(M))
    filters, _ = _dht_matrices(int(M))
    output = np.zeros_like(image)
    bits: list[bool] = []
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
            coeff = _rotate_block(coeff)
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
        else:
            if rotated_mode:
                # blkqtr keeps the first column and the stored angle y(1,2),
                # and discards every remaining directional coefficient.
                coded = np.zeros_like(coeff)
                coded[:, 0] = coeff[:, 0]
                coded[0, 1] = coeff[0, 1]
                output[top:bottom, left:right] = coded
            else:
                output[top:bottom, left:right] = _matlab_round(255 * coeff)

    block = 2 ** (M + 1)
    for row in range(image.shape[0] // block):
        for col in range(image.shape[1] // block):
            visit(row, col, int(M))
    return output, np.asarray(bits, dtype=bool)


def idhtqt(Z: Array, qtbits: Sequence[bool], M: int, flag: str = "n") -> Array:
    """Reconstruct an image from DHT-quadtree coefficients."""

    coded = _validate_image(Z, int(M))
    filters, windows = _dht_matrices(int(M))
    output = np.zeros_like(coded)
    weight = np.zeros_like(coded)
    reader = _BitReader(list(np.asarray(qtbits, dtype=bool).ravel()))
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
            coeff = _rotate_block(coeff, inverse=True)
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


def im2qtb(Z: Array, qtbits: Sequence[bool], M=4) -> list[Array]:
    """Convert a quadtree coefficient image to blocks grouped by level."""

    image = np.asarray(Z)
    if isinstance(M, (list, tuple)):
        blocks = [np.asarray(v).copy() for v in M]
        levels = len(blocks)
    else:
        levels = int(M)
        blocks = [np.empty((2 ** (level + 2), 2 ** (level + 2), 0), dtype=image.dtype) for level in range(levels)]
    _validate_image(image, levels)
    reader = _BitReader(list(np.asarray(qtbits, dtype=bool).ravel()))

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


def qtb2im(blocks: Sequence[Array], qtbits: Sequence[bool], xsiz: Sequence[int], M: int | None = None) -> Array:
    """Restore a quadtree coefficient image from blocks grouped by level."""

    levels = len(blocks) if M is None else int(M)
    queues = [list(np.moveaxis(np.asarray(value), 2, 0)) for value in blocks]
    output = np.zeros(tuple(int(v) for v in xsiz[:2]), dtype=np.result_type(*[np.asarray(v).dtype for v in blocks]))
    reader = _BitReader(list(np.asarray(qtbits, dtype=bool).ravel()))

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


def qtplot(X: Array, qtbits: Sequence[bool], M: int = 4, circle: bool = False, *plot_args, ax=None):
    """Plot or return quadtree boundaries/circles."""

    image = _validate_image(X, int(M))
    reader = _BitReader(list(np.asarray(qtbits, dtype=bool).ravel()))
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
