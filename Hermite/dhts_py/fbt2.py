"""Direct translation of ``fbt2.m``."""

from .fbt import fbt


def fbt2(Y, N: int):
    return fbt(fbt(Y, N, 2), N, 1)


__all__ = ["fbt2"]
