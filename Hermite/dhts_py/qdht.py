"""Direct translation of ``qdht.m``."""

import numpy as np


def qdht(Y, tdir: str = "fwd", cod: str = "", N: int | None = None, D: int | None = None, *args):
    out = np.asarray(Y, dtype=np.float64).copy()
    odd_rows, odd_cols = np.arange(0, out.shape[0], 2), np.arange(0, out.shape[1], 2)
    even_rows, even_cols = np.arange(1, out.shape[0], 2), np.arange(1, out.shape[1], 2)
    rejected_a = np.ix_(even_rows, odd_cols)
    rejected_b = np.ix_(odd_rows, even_cols)
    if tdir.lower() == "fwd":
        start = 1 if any(ch in cod for ch in "lhL") else 0
        out[rejected_a + (slice(start, None),)] = np.nan
        out[rejected_b + (slice(start, None),)] = np.nan
    elif tdir.lower() == "inv":
        out[rejected_a + (slice(None),)] = 0
        out[rejected_b + (slice(None),)] = 0
    else:
        raise ValueError("tdir must be 'fwd' or 'inv'")
    if not cod:
        return out
    if N is None or D is None:
        raise ValueError("N and D are required when cod is supplied")
    key = cod[0].lower()
    for rows, cols in ((odd_rows, odd_cols), (even_rows, even_cols)):
        sub = out[np.ix_(rows, cols, np.arange(out.shape[2]))]
        local_args = list(args)
        for i, value in enumerate(local_args):
            if isinstance(value, np.ndarray) and value.shape[:2] == out.shape[:2]:
                local_args[i] = value[np.ix_(rows, cols)]
        if key == "r":
            from .rdht import rdht

            sub = rdht(sub, N, D, tdir, *(local_args or [None]))
        elif key == "d":
            from .ddht import ddht

            sub = ddht(sub, N, D, tdir, *(local_args or [None]))
        elif key in {"s", "c"}:
            from .sdht2 import sdht2

            sub = sdht2(sub, N, D, *(local_args if key == "s" else [None, None]))[0]
        out[np.ix_(rows, cols, np.arange(sub.shape[2]))] = sub
    return out


__all__ = ["qdht"]
