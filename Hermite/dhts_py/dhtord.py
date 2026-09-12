"""Selección y orden de los coeficientes bidimensionales de Hermite.

Este módulo es la única fuente de verdad para asociar cada canal del stack
con su par ``(m, n)``. La función pública es :func:`dhtord`.
"""

import numpy as np


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} debe ser un entero >= 0.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} debe ser un entero >= 0.")
    return value


def dhtord(
    N: int,
    D: int,
    coefficient_region: str = "triangle",
) -> list[tuple[int, int]]:
    """Devuelve los pares ``(m, n)`` en el orden de los canales.

    Parameters
    ----------
    N : int
        Escala del filter bank. Los órdenes individuales satisfacen
        ``0 <= m, n <= N``.
    D : int
        En ``"triangle"`` es el máximo orden total ``m + n`` y debe cumplir
        ``0 <= D <= 2*N``. En ``"square"`` es el máximo por eje y se aplica
        ``M = min(D, N)``.
    coefficient_region : {"triangle", "square"}, default="triangle"
        Región de coeficientes que se conserva.

    Returns
    -------
    orders : list of tuple of int
        Pares ``(m, n)`` agrupados por orden total. Dentro de cada grupo, ``m``
        decrece y ``n`` crece. Por tanto los primeros canales son ``L00``,
        ``L10``, ``L01``, ``L20``, ``L11`` y ``L02``.
    """

    N = _nonnegative_integer("N", N)
    D = _nonnegative_integer("D", D)
    if not isinstance(coefficient_region, str) or coefficient_region not in {
        "triangle",
        "square",
    }:
        raise ValueError('coefficient_region debe ser "triangle" o "square".')

    if coefficient_region == "triangle":
        if D > 2 * N:
            raise ValueError("Para triangle, D debe cumplir 0 <= D <= 2*N.")
        max_axis_order = N
        max_total_order = D
    else:
        max_axis_order = min(D, N)
        max_total_order = 2 * max_axis_order

    orders: list[tuple[int, int]] = []
    for total in range(max_total_order + 1):
        largest_m = min(max_axis_order, total)
        smallest_m = max(0, total - max_axis_order)
        orders.extend(
            (m, total - m) for m in range(largest_m, smallest_m - 1, -1)
        )
    return orders


__all__ = ["dhtord"]
