"""Private shared machinery for the quadtree translations.

The public algorithms live in their MATLAB-homologous modules.  These helper
objects are shared because the five quadtree routines traverse the same bit
stream and use the same block transform/rotation representation.
"""

from dataclasses import dataclass

import numpy as np

from .dhtord import dhtord
from .rdht import rdht


def matlab_round(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def dht_matrices(M: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
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


def matrix_to_stack(matrix: np.ndarray) -> np.ndarray:
    T = matrix.shape[0]
    orders = dhtord(T - 1, 2 * (T - 1), 2)
    return np.asarray([matrix[v, h] for h, v in orders], dtype=float)[None, None, :]


def stack_to_matrix(stack: np.ndarray, T: int) -> np.ndarray:
    matrix = np.zeros((T, T), dtype=float)
    for value, (horizontal, vertical) in zip(np.asarray(stack).ravel(), dhtord(T - 1, 2 * (T - 1), 2)):
        matrix[vertical, horizontal] = value
    return matrix


def rotate_block(matrix: np.ndarray, inverse: bool = False) -> np.ndarray:
    T = matrix.shape[0]
    stack = matrix_to_stack(matrix)
    if inverse:
        angle = 2 * np.pi * matrix[0, 1]
        index = np.flatnonzero((dhtord(T - 1, 2 * (T - 1), 2) == (1, 0)).all(axis=1))
        stack[..., list(index)] = 0
        rotated = rdht(stack, T - 1, 2 * (T - 1), "inv", -angle)
    else:
        angle = np.arctan2(matrix[0, 1], matrix[1, 0])
        rotated = rdht(stack, T - 1, 2 * (T - 1), "fwd", -angle)
    result = stack_to_matrix(rotated, T)
    if not inverse:
        result[0, 1] = angle / (2 * np.pi)
    return result


@dataclass
class BitReader:
    bits: list[bool]
    position: int = 0

    def read(self) -> bool:
        if self.position >= len(self.bits):
            raise ValueError("qtbits ended before the quadtree traversal completed")
        value = bool(self.bits[self.position])
        self.position += 1
        return value


def validate_image(image: np.ndarray, M: int) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError("quadtree functions require a 2-D array")
    block = 2 ** (M + 1)
    if arr.shape[0] % block or arr.shape[1] % block:
        raise ValueError(f"image dimensions must be multiples of {block} for M={M}")
    return arr
