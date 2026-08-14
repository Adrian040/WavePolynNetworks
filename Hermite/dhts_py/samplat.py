"""Direct translation of ``samplat.m``."""

import numpy as np


def samplat(p, N, T=None, shape: str = "full"):
    positions = np.asarray(p)
    multiscale = T is None or isinstance(T, str)
    if isinstance(T, str):
        shape = T
    if multiscale:
        values = np.atleast_1d(N).astype(int)
        if values.size > 1:
            base_n, levels = int(values[0]), int(values[1])
        else:
            base_n, levels = 8, int(values[0])
        result = [samplat(positions, base_n, int(round(np.sqrt(base_n / 2))), shape)]
        for _ in range(1, levels):
            result.append(samplat(result[-1], 6, 2, shape))
        return result
    N, T = int(N), int(T)
    step = 1 if positions.size < 2 else positions[1] - positions[0]
    key = shape.lower()
    if key == "full":
        left = positions[0] - step * (N / 2 - np.arange(N // 2))
        right = positions[-1] + step * np.arange(1, N // 2 + 1)
        positions = np.r_[left, positions, right]
    elif key == "valid":
        positions = positions[N // 2 : positions.size - N // 2]
    elif key not in {"same", "symm", "repeat", "asymm", "cyclic"}:
        raise ValueError(f"invalid shape {shape!r}")
    return positions[::T]


__all__ = ["samplat"]
