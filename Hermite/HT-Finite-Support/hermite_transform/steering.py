"""Estimación de orientación y steering de coeficientes Hermite."""

from collections.abc import Mapping, Sequence
from math import comb

import numpy as np


Array = np.ndarray
Order = tuple[int, int]


def dominant_theta(coefficients: Mapping[Order, Array]) -> Array:
    """Estima la dirección local del gradiente.

    Parameters
    ----------
    coefficients : mapping
        Mapas que contienen ``L_10`` y ``L_01``.

    Returns
    -------
    ndarray
        ``atan2(L_01, L_10)`` en radianes. Es la dirección normal al borde,
        no su dirección tangente.
    """
    if (1, 0) not in coefficients or (0, 1) not in coefficients:
        raise ValueError("La orientación dominante requiere L_10 y L_01.")
    return np.arctan2(coefficients[(0, 1)], coefficients[(1, 0)])


def angle_map(theta: float | Array, shape: tuple[int, int]) -> Array:
    """Expande un ángulo escalar o mapa a la forma de los coeficientes.

    Parameters
    ----------
    theta : float or ndarray
        Ángulo en radianes.
    shape : tuple of int
        Forma espacial requerida.

    Returns
    -------
    ndarray
        Mapa ``float64`` de ángulos finitos.
    """
    theta = np.asarray(theta, dtype=np.float64)
    if not np.isfinite(theta).all():
        raise ValueError("theta debe contener valores finitos.")
    try:
        return np.broadcast_to(theta, shape).astype(np.float64, copy=False)
    except ValueError as exc:
        raise ValueError(f"theta no es compatible con la forma {shape}.") from exc


def rotate_block(block: Array, theta: float | Array) -> Array:
    """Rota un bloque completo de igual orden total.

    Parameters
    ----------
    block : ndarray, shape (..., degree + 1)
        Coeficientes ``[L_r0, L_(r-1)1, ..., L_0r]``.
    theta : float or ndarray
        Ángulo de steering en radianes.

    Returns
    -------
    ndarray
        Bloque rotado con la misma forma y normalización Hermite.
    """
    block = np.asarray(block, dtype=np.float64)
    degree = block.shape[-1] - 1
    if degree == 0:
        return block.copy()

    theta = angle_map(theta, block.shape[:-1])
    cosine, sine = np.cos(theta), np.sin(theta)
    pascal = np.sqrt([comb(degree, index) for index in range(degree + 1)])
    work = block.copy()
    if degree > 1:
        work[..., 1:degree] /= pascal[1:degree]

    rotated = np.empty_like(work)
    active = degree + 1
    for output_index in range(degree):
        reduced = work.copy()
        length = active
        for _ in range(output_index, degree):
            reduced = (
                cosine[..., None] * reduced[..., : length - 1]
                + sine[..., None] * reduced[..., 1:length]
            )
            length -= 1
        rotated[..., output_index] = reduced[..., 0] * pascal[output_index]
        work = (
            cosine[..., None] * work[..., 1:active]
            - sine[..., None] * work[..., : active - 1]
        )
        active -= 1

    rotated[..., degree] = work[..., 0]
    return rotated


def rotate_coefficients(
    coefficients: Mapping[Order, Array],
    theta: float | Array,
    orders: Sequence[Order],
) -> dict[Order, Array]:
    """Rota los coeficientes por bloques completos de grado total.

    Parameters
    ----------
    coefficients : mapping
        Mapas cartesianos, incluidos los auxiliares requeridos.
    theta : float or ndarray
        Ángulo en radianes.
    orders : sequence of tuple of int
        Conjunto completo de órdenes que se rotará.

    Returns
    -------
    dict
        Coeficientes rotados con las mismas claves.
    """
    orders = list(orders)
    missing = [order for order in orders if order not in coefficients]
    if missing:
        raise ValueError(f"Faltan coeficientes: {missing[:5]}.")
    shape = np.asarray(coefficients[orders[0]]).shape
    theta = angle_map(theta, shape)
    result = {order: np.asarray(coefficients[order], dtype=np.float64).copy()
              for order in orders}
    order_set = set(orders)

    for total in sorted({m + n for m, n in orders}):
        block_orders = [(m, total - m) for m in range(total, -1, -1)]
        missing_block = [order for order in block_orders if order not in order_set]
        if missing_block:
            raise ValueError(
                f"El bloque de grado {total} está incompleto: {missing_block}."
            )
        block = np.stack([result[order] for order in block_orders], axis=-1)
        rotated = rotate_block(block, theta)
        for index, order in enumerate(block_orders):
            result[order] = rotated[..., index]
    return result


def inverse_rotate_coefficients(
    coefficients: Mapping[Order, Array],
    theta: float | Array,
    orders: Sequence[Order],
) -> dict[Order, Array]:
    """Deshace el steering aplicando el mismo operador con ``-theta``.

    Parameters
    ----------
    coefficients : mapping
        Coeficientes rotados.
    theta : float or ndarray
        Ángulo directo en radianes.
    orders : sequence of tuple of int
        Órdenes completos que se recuperarán.

    Returns
    -------
    dict
        Coeficientes cartesianos recuperados.
    """
    return rotate_coefficients(coefficients, -np.asarray(theta), orders)
