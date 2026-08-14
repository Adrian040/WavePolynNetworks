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


def _symmetric_indices(length: int, before: int, after: int) -> np.ndarray:
    positions = np.arange(-before, length + after, dtype=int)
    period = 2 * length
    folded = np.mod(positions, period)
    return np.where(folded < length, folded, period - 1 - folded)


def _synthesis_geometry(
    coefficients: np.ndarray,
    target: tuple[int, int],
    N: int,
    T: int,
    shape: str,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray, np.ndarray]:
    expanded_shape = (target[0] + N, target[1] + N)
    if shape == "full":
        first_sample = 0
        grid = coefficients
    else:
        extension_width = (N + 1) // 2
        before = extension_width // T
        first_sample = extension_width - T * before
        grid_shape = (
            len(range(first_sample, expanded_shape[0], T)),
            len(range(first_sample, expanded_shape[1], T)),
        )
        after = (
            grid_shape[0] - coefficients.shape[0] - before,
            grid_shape[1] - coefficients.shape[1] - before,
        )
        if min(after) < 0:
            raise ValueError("El stack no es compatible con image_shape, N, T y shape='symm'.")

        rows = _symmetric_indices(coefficients.shape[0], before, after[0])
        columns = _symmetric_indices(coefficients.shape[1], before, after[1])
        grid = np.zeros(grid_shape + (coefficients.shape[2],), dtype=np.float64)

        # Sólo L00 se prolonga en el borde. Los canales de orden positivo se
        # conservan en la región analizada y se anulan en la extensión dual.
        grid[..., 0] = coefficients[..., 0][np.ix_(rows, columns)]
        if coefficients.shape[2] > 1:
            row_end = grid_shape[0] - after[0] if after[0] else grid_shape[0]
            column_end = grid_shape[1] - after[1] if after[1] else grid_shape[1]
            grid[before:row_end, before:column_end, 1:] = coefficients[..., 1:]

    sampled_rows = np.arange(first_sample, expanded_shape[0], T)
    sampled_columns = np.arange(first_sample, expanded_shape[1], T)
    expected_spatial_shape = (sampled_rows.size, sampled_columns.size)
    if grid.shape[:2] != expected_spatial_shape:
        raise ValueError(
            "El shape espacial de coefficients no coincide con image_shape, N, T y shape: "
            f"se esperaba {expected_spatial_shape} y se recibió {grid.shape[:2]}."
        )
    return grid, expanded_shape, sampled_rows, sampled_columns


def idht2(
    coefficients: np.ndarray,
    image_shape: tuple[int, int],
    N: int,
    D: int,
    T: int = 1,
    coefficient_region: str = "triangle",
    shape: str = "full",
) -> np.ndarray:
    """Reconstruye una imagen desde coefficient maps cartesianos.

    Parameters
    ----------
    coefficients : ndarray, shape (sampled_rows, sampled_columns, C)
        Stack producido por :func:`dht2` con los mismos ``N``, ``D``, ``T``,
        ``coefficient_region`` y ``shape``.
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
    shape : {"full", "symm"}, default="full"
        Geometría usada en el análisis. Debe coincidir con el valor pasado a
        :func:`dht2`.

    Returns
    -------
    reconstruction : ndarray, shape image_shape
        Imagen reconstruida en ``float64``.

    Notes
    -----
    Una región triangular truncada produce una aproximación y no se fuerza a
    ser exacta. El square completo ``D=N`` emplea los ``(N+1)^2`` coefficient
    maps del banco y conserva la síntesis completa para el sampling indicado.
    """

    target = _image_shape(image_shape)
    T = _sampling_step(T)
    orders = dhtord(N, D, coefficient_region)
    N = int(N)
    if not isinstance(shape, str) or shape.lower() not in {"full", "symm"}:
        raise ValueError('shape debe ser "full" o "symm".')
    shape = shape.lower()
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
    coefficient_grid, expanded_shape, sampled_rows, sampled_columns = _synthesis_geometry(
        values, target, N, T, shape
    )

    reconstruction = np.zeros(target, dtype=np.float64)
    for channel, (m, n) in enumerate(orders):
        expanded = np.zeros(expanded_shape, dtype=np.float64)
        expanded[np.ix_(sampled_rows, sampled_columns)] = coefficient_grid[..., channel]
        kernel = np.outer(synthesis_filters[:, n], synthesis_filters[:, m])
        reconstruction += convolve2d(expanded, kernel, mode="valid")
    return reconstruction


__all__ = ["idht2"]
