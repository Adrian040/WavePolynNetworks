"""Transformada de Hermite directional/multidirectional.

La DDHT evalúa cada bloque cartesiano de orden total en varias direcciones.
Se mantiene separada de la RDHT, que expresa el mismo bloque en un único
sistema de referencia rotado. La función pública es :func:`ddht`.
"""

from math import comb

import numpy as np

from .dhtord import dhtord


def _theta_map(theta: float | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    angle = np.asarray(theta, dtype=np.float64)
    if angle.ndim == 0:
        return np.full(shape, float(angle), dtype=np.float64)
    try:
        return np.broadcast_to(angle, shape).astype(np.float64, copy=False)
    except ValueError as exc:
        raise ValueError(f"theta tiene shape {angle.shape}; se esperaba {shape}.") from exc


def ddht(
    coefficients: np.ndarray,
    theta: float | np.ndarray,
    N: int,
    D: int,
    direction: str = "forward",
    coefficient_region: str = "triangle",
) -> np.ndarray:
    """Calcula la DDHT directa o inversa sobre bloques triangulares completos.

    Parameters
    ----------
    coefficients : ndarray, shape (rows, columns, C)
        Coefficient maps en el orden definido por :func:`dhtord`.
    theta : float or ndarray, shape (rows, columns)
        Orientación base en radianes.
    N : int
        Escala del filter bank asociado.
    D : int
        Máximo orden total. La implementación mínima requiere ``D <= N`` para
        que todos los bloques de órdenes ``0`` a ``D`` estén completos.
    direction : {"forward", "inverse"}, default="forward"
        Sentido de la transformación. También acepta ``"fwd"`` e ``"inv"``.
    coefficient_region : {"triangle"}, default="triangle"
        La DDHT square no se admite porque los squares parciales contienen
        bloques incompletos y requieren una formulación adicional.

    Returns
    -------
    transformed : ndarray
        Stack ``float64`` con el mismo shape que la entrada.
    """

    directions = {"forward": "forward", "fwd": "forward", "inverse": "inverse", "inv": "inverse"}
    if not isinstance(direction, str) or direction.lower() not in directions:
        raise ValueError('direction debe ser "forward" o "inverse".')
    if coefficient_region != "triangle":
        raise ValueError('DDHT sólo admite coefficient_region="triangle".')

    orders = dhtord(N, D, coefficient_region)
    if int(D) > int(N):
        raise ValueError("DDHT requiere D <= N para utilizar bloques de orden total completos.")

    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("coefficients debe ser un stack 3-D (rows, columns, channels).")
    if values.shape[-1] != len(orders):
        raise ValueError(
            f"Se esperaban {len(orders)} canales según dhtord; se recibieron "
            f"{values.shape[-1]}."
        )

    angle = _theta_map(theta, values.shape[:2])
    result = values.copy()
    for degree in range(1, int(D) + 1):
        indices = [index for index, order in enumerate(orders) if sum(order) == degree]
        block = values[..., indices]
        powers = np.arange(degree + 1)
        normalization = np.sqrt(
            np.asarray([comb(degree, power) for power in powers], dtype=np.float64)
        )
        sampled_directions = angle[..., None] + np.arange(degree + 1) * np.pi / (degree + 1)
        basis = (
            np.cos(sampled_directions)[..., None] ** (degree - powers)
            * np.sin(sampled_directions)[..., None] ** powers
        )
        transform_matrix = basis * normalization

        if directions[direction.lower()] == "forward":
            result[..., indices] = np.einsum("...jm,...m->...j", transform_matrix, block)
        else:
            result[..., indices] = np.linalg.solve(
                transform_matrix,
                block[..., None],
            )[..., 0]
    return result


__all__ = ["ddht"]
