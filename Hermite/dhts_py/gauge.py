"""Estimación de orientación local a partir de coeficientes de Hermite.

Este módulo obtiene el mapa angular ``theta`` mediante una condición basada
en gradient o Hessian. La función pública es :func:`gauge`.
"""

from math import comb

import numpy as np

from .dhtord import dhtord


def _binomial_norm(degree: int) -> np.ndarray:
    return np.sqrt(
        np.asarray([comb(degree, index) for index in range(degree + 1)], dtype=np.float64)
    )


def gauge(
    coefficients: np.ndarray,
    N: int,
    D: int,
    mode: str = "gradient",
    coefficient_region: str = "triangle",
) -> np.ndarray:
    """Calcula la orientación local de un stack DHT cartesiano.

    Parameters
    ----------
    coefficients : ndarray, shape (rows, columns, C)
        Coefficient maps cartesianos en el orden dado por :func:`dhtord`.
    N : int
        Escala del filter bank usado para calcular los coeficientes.
    D : int
        Límite de órdenes, interpretado según ``coefficient_region``.
    mode : {"gradient", "hessian"}, default="gradient"
        ``"gradient"`` usa el bloque de primer orden. ``"hessian"`` usa el
        bloque de segundo orden y el gradient para resolver la ambigüedad de
        dirección. También se aceptan los alias ``"grad"`` y ``"hess"``.
    coefficient_region : {"triangle", "square"}, default="triangle"
        Región con la que se creó el stack.

    Returns
    -------
    theta : ndarray, shape (rows, columns)
        Orientación en radianes y ``float64``.
    """

    aliases = {
        "gradient": 1,
        "grad": 1,
        "hessian": 2,
        "hess": 2,
    }
    if not isinstance(mode, str) or mode.lower() not in aliases:
        raise ValueError('mode debe ser "gradient" o "hessian".')
    degree = aliases[mode.lower()]

    orders = dhtord(N, D, coefficient_region)
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("coefficients debe ser un stack 3-D (rows, columns, channels).")
    if values.shape[-1] != len(orders):
        raise ValueError(
            f"Se esperaban {len(orders)} canales según dhtord; se recibieron "
            f"{values.shape[-1]}."
        )

    selected = [index for index, order in enumerate(orders) if sum(order) == degree]
    if len(selected) != degree + 1:
        raise ValueError(f"No está disponible el bloque completo de orden total {degree}.")

    block = values[..., selected]
    normalization = _binomial_norm(degree)
    odd_weights = normalization[1::2] * (-1.0) ** np.arange(normalization[1::2].size)
    even_weights = normalization[0::2] * (-1.0) ** np.arange(normalization[0::2].size)
    odd_component = np.tensordot(block[..., 1::2], odd_weights, axes=([-1], [0]))
    even_component = np.tensordot(block[..., 0::2], even_weights, axes=([-1], [0]))

    theta = np.arctan2(odd_component, even_component) / degree
    if degree == 1:
        return theta

    # El Hessian define un eje, no una dirección. Se comparan sus respuestas
    # equivalentes y el gradient fija el sentido cuando tiene magnitud no nula.
    powers = np.arange(degree + 1)
    increments = np.arange(degree + 1) * np.pi / degree
    responses = np.stack(
        [
            np.sum(
                block
                * normalization
                * np.cos(theta[..., None] + increment) ** (degree - powers)
                * np.sin(theta[..., None] + increment) ** powers,
                axis=-1,
            )
            for increment in increments
        ],
        axis=-1,
    )
    theta = theta + np.argmax(np.abs(responses), axis=-1) * (np.pi / degree)

    index = {order: channel for channel, order in enumerate(orders)}
    gradient_projection = (
        np.cos(theta) * values[..., index[(1, 0)]]
        + np.sin(theta) * values[..., index[(0, 1)]]
    )
    return np.where(gradient_projection < 0.0, theta - np.pi, theta)


__all__ = ["gauge"]
