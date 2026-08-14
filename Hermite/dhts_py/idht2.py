"""Reconstrucción bidimensional mediante la DHT inversa.

Este módulo sintetiza una imagen desde exactamente los coefficient maps
seleccionados por :func:`dhtord`, incluidos todos los términos de una región
cuadrada. La función pública es :func:`idht2`.
"""

import numpy as np
from scipy.signal import convolve2d

from .dhtmtx import dhtmtx
from .dhtord import dhtord


def _sampling_step(T: int) -> int:
    if isinstance(T, (bool, np.bool_)) or not isinstance(T, (int, np.integer)) or int(T) < 1:
        raise ValueError("T debe ser un entero >= 1.")
    return int(T)


def _image_shape(image_shape) -> tuple[int, int]:
    if isinstance(image_shape, (str, bytes)):
        raise ValueError("image_shape debe contener exactamente dos enteros positivos.")
    try:
        dimensions = tuple(image_shape)
    except TypeError as exc:
        raise ValueError("image_shape debe contener exactamente dos enteros positivos.") from exc
    if len(dimensions) != 2:
        raise ValueError("image_shape debe contener exactamente dos enteros positivos.")
    result = []
    for value in dimensions:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 1
        ):
            raise ValueError("image_shape debe contener exactamente dos enteros positivos.")
        result.append(int(value))
    return result[0], result[1]


def idht2(
    coefficients: np.ndarray,
    image_shape: tuple[int, int],
    N: int,
    D: int,
    T: int = 1,
    coefficient_region: str = "triangle",
) -> np.ndarray:
    """Reconstruye una imagen desde coefficient maps cartesianos.

    Parameters
    ----------
    coefficients : ndarray, shape (sampled_rows, sampled_columns, C)
        Stack producido por :func:`dht2` con los mismos ``N``, ``D``, ``T`` y
        ``coefficient_region``.
    image_shape : sequence of two int
        Shape ``(rows, columns)`` de la imagen antes del análisis.
    N : int
        Escala del filter bank; sus filtros tienen longitud ``N + 1``.
    D : int
        Para ``"triangle"`` es el máximo orden total. Para ``"square"`` es el
        máximo orden por eje, limitado por ``N``.
    T : int, default=1
        Paso de sampling usado en el análisis. Debe ser un entero >= 1.
    coefficient_region : {"triangle", "square"}, default="triangle"
        Región que determina exactamente qué mapas participan en la suma de
        síntesis.

    Returns
    -------
    reconstruction : ndarray, shape image_shape
        Imagen reconstruida en ``float64``.

    Notes
    -----
    Una región triangular truncada produce una aproximación y no se fuerza a
    ser exacta. Con ``T=1`` y el square completo ``D=N`` se emplean los
    ``(N+1)^2`` coefficient maps del banco.
    """

    target = _image_shape(image_shape)
    T = _sampling_step(T)
    orders = dhtord(N, D, coefficient_region)
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("coefficients debe ser un stack 3-D (rows, columns, channels).")
    if values.shape[-1] != len(orders):
        raise ValueError(
            f"Se esperaban {len(orders)} canales según dhtord; se recibieron "
            f"{values.shape[-1]}."
        )

    max_individual_order = max(max(order) for order in orders)
    _, synthesis_filters = dhtmtx(N, max_individual_order, T)
    expanded_shape = (target[0] + N, target[1] + N)
    sampled_rows = np.arange(0, expanded_shape[0], T)
    sampled_columns = np.arange(0, expanded_shape[1], T)
    expected_spatial_shape = (sampled_rows.size, sampled_columns.size)
    if values.shape[:2] != expected_spatial_shape:
        raise ValueError(
            "El shape espacial de coefficients no coincide con image_shape, N y T: "
            f"se esperaba {expected_spatial_shape} y se recibió {values.shape[:2]}."
        )

    reconstruction = np.zeros(target, dtype=np.float64)
    for channel, (m, n) in enumerate(orders):
        expanded = np.zeros(expanded_shape, dtype=np.float64)
        expanded[np.ix_(sampled_rows, sampled_columns)] = values[..., channel]
        kernel = np.outer(synthesis_filters[:, n], synthesis_filters[:, m])
        reconstruction += convolve2d(expanded, kernel, mode="valid")
    return reconstruction


__all__ = ["idht2"]
