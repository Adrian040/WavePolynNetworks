"""Transformada de Hermite rotada y su operación inversa.

El steering se aplica por bloques completos de igual orden total y conserva el
orden de canales definido por :func:`dhtord`. La función pública es
:func:`rdht`.
"""

from math import comb

import numpy as np

from .dhtord import dhtord


def _binomial_norm(degree: int) -> np.ndarray:
    return np.sqrt(
        np.asarray([comb(degree, index) for index in range(degree + 1)], dtype=np.float64)
    )


def _theta_map(theta: float | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    angle = np.asarray(theta, dtype=np.float64)
    if angle.ndim == 0:
        return np.full(shape, float(angle), dtype=np.float64)
    try:
        return np.broadcast_to(angle, shape).astype(np.float64, copy=False)
    except ValueError as exc:
        raise ValueError(f"theta tiene shape {angle.shape}; se esperaba {shape}.") from exc


def _rotate_block(block: np.ndarray, theta: np.ndarray) -> np.ndarray:
    degree = block.shape[-1] - 1
    if degree <= 0:
        return block.copy()

    cosine = np.cos(theta)
    sine = np.sin(theta)
    normalization = _binomial_norm(degree)
    work = np.asarray(block, dtype=np.float64).copy()
    if degree > 1:
        work[..., 1:degree] /= normalization[1:degree]

    rotated = np.empty_like(work)
    active_length = degree + 1
    for output_order in range(degree):
        reduced = work.copy()
        reduced_length = active_length
        for _ in range(output_order, degree):
            reduced = (
                cosine[..., None] * reduced[..., : reduced_length - 1]
                + sine[..., None] * reduced[..., 1:reduced_length]
            )
            reduced_length -= 1
        rotated[..., output_order] = reduced[..., 0] * normalization[output_order]
        work = (
            cosine[..., None] * work[..., 1:active_length]
            - sine[..., None] * work[..., : active_length - 1]
        )
        active_length -= 1
    rotated[..., degree] = work[..., 0]
    return rotated


def rdht(
    coefficients: np.ndarray,
    theta: float | np.ndarray,
    N: int,
    D: int,
    direction: str = "forward",
    coefficient_region: str = "triangle",
) -> np.ndarray:
    """Aplica steering directo o inverso a coefficient maps de Hermite.

    Parameters
    ----------
    coefficients : ndarray, shape (rows, columns, C)
        Stack en el orden definido por :func:`dhtord`.
    theta : float or ndarray, shape (rows, columns)
        Ángulo de steering en radianes. Un escalar se aplica a todo el stack.
    N : int
        Escala del filter bank asociado.
    D : int
        Límite de órdenes, interpretado según ``coefficient_region``.
    direction : {"forward", "inverse"}, default="forward"
        Sentido de la transformación. Se aceptan ``"fwd"`` e ``"inv"`` como
        alias.
    coefficient_region : {"triangle", "square"}, default="triangle"
        Región usada para crear el stack. Cada bloque disponible se obtiene de
        :func:`dhtord`. El modo square sólo admite la base completa ``D=N``;
        un square parcial no contiene bloques cerrados bajo steering.

    Returns
    -------
    rotated : ndarray
        Stack ``float64`` con el mismo shape y orden de canales que la entrada.
    """

    directions = {
        "forward": 1.0,
        "fwd": 1.0,
        "inverse": -1.0,
        "inv": -1.0,
    }
    if not isinstance(direction, str) or direction.lower() not in directions:
        raise ValueError('direction debe ser "forward" o "inverse".')

    orders = dhtord(N, D, coefficient_region)
    if coefficient_region == "square" and int(D) != int(N):
        raise ValueError(
            "RDHT sobre un square parcial no contiene todos los componentes "
            "necesarios de algunos órdenes totales. Use coefficient_region="
            '"triangle" para steering truncado por orden total o use el square '
            "completo D=N."
        )
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("coefficients debe ser un stack 3-D (rows, columns, channels).")
    if values.shape[-1] != len(orders):
        raise ValueError(
            f"Se esperaban {len(orders)} canales según dhtord; se recibieron "
            f"{values.shape[-1]}."
        )

    angle = directions[direction.lower()] * _theta_map(theta, values.shape[:2])
    result = values.copy()
    totals = sorted({sum(order) for order in orders})
    for total in totals:
        indices = [index for index, order in enumerate(orders) if sum(order) == total]
        result[..., indices] = _rotate_block(values[..., indices], angle)
    return result


__all__ = ["rdht"]
