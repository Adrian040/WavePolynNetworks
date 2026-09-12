"""Síntesis truncada de los coeficientes cartesianos de Hermite."""

from collections.abc import Mapping

import numpy as np
from scipy.ndimage import convolve1d


Array = np.ndarray
Order = tuple[int, int]


def synthesize(
    coefficients: Mapping[Order, Array],
    bank: Mapping,
    image_shape: tuple[int, int],
    sampling_step: int = 1,
) -> Array:
    """Reconstruye mediante overlap-add y normalización por la ventana.

    Parameters
    ----------
    coefficients : mapping
        Coeficientes cartesianos de los órdenes públicos del banco.
    bank : mapping
        Banco usado durante el análisis.
    image_shape : tuple of int
        Forma ``(alto, ancho)`` de salida.
    sampling_step : int, default=1
        Separación de la retícula de análisis.

    Returns
    -------
    ndarray
        Reconstrucción truncada ``float64`` con la forma solicitada.

    Notes
    -----
    El numerador usa los mismos filtros Hermite--Gaussianos del análisis. El
    denominador suma las ventanas de orden cero, equivalentes a ``w²`` en esta
    convención. No es una inversa exacta de una expansión infinita.
    """
    if isinstance(sampling_step, (bool, np.bool_)) or not isinstance(
        sampling_step, (int, np.integer)
    ) or sampling_step < 1:
        raise ValueError("sampling_step debe ser un entero positivo.")
    if len(image_shape) != 2 or min(image_shape) <= 0:
        raise ValueError("image_shape debe contener dos tamaños positivos.")

    height, width = map(int, image_shape)
    rows = np.arange(0, height, sampling_step)
    columns = np.arange(0, width, sampling_step)
    expected_shape = (rows.size, columns.size)
    numerator = np.zeros((height, width), dtype=np.float64)

    for m, n in bank["orders"]:
        if (m, n) not in coefficients:
            raise KeyError(f"Falta el coeficiente {(m, n)} para la síntesis.")
        coefficient_map = np.asarray(coefficients[(m, n)], dtype=np.float64)
        if coefficient_map.shape != expected_shape:
            raise ValueError(
                f"L_{m}{n} tiene forma {coefficient_map.shape}; "
                f"se esperaba {expected_shape}."
            )
        upsampled = np.zeros((height, width), dtype=np.float64)
        upsampled[np.ix_(rows, columns)] = coefficient_map
        horizontal = convolve1d(
            upsampled,
            bank["filters"][m],
            axis=1,
            mode="constant",
        )
        numerator += convolve1d(
            horizontal,
            bank["filters"][n],
            axis=0,
            mode="constant",
        )

    mask = np.zeros((height, width), dtype=np.float64)
    mask[np.ix_(rows, columns)] = 1.0
    denominator = convolve1d(
        mask,
        bank["filters"][0],
        axis=1,
        mode="constant",
    )
    denominator = convolve1d(
        denominator,
        bank["filters"][0],
        axis=0,
        mode="constant",
    )
    if np.any(denominator <= 0):
        raise ValueError("sampling_step deja píxeles sin cobertura.")
    return numerator / denominator
