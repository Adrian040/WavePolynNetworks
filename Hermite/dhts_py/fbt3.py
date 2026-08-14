"""Direct translation of ``fbt3.m``."""

from .fbt import fbt


def fbt3(Y, N: int):
    return fbt(fbt(fbt(Y, N, 3), N, 2), N, 1)


__all__ = ["fbt3"]
