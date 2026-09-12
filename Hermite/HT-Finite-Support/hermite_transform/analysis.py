"""Análisis cartesiano Hermite--Gaussiano mediante filtros separables."""

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.ndimage import correlate1d


Array = np.ndarray
Order = tuple[int, int]

_BOUNDARIES = {
    "symmetric": ("symmetric", "reflect"),
    "symm": ("symmetric", "reflect"),
    "reflect": ("symmetric", "reflect"),
    "edge": ("edge", "nearest"),
    "nearest": ("edge", "nearest"),
    "replicate": ("edge", "nearest"),
    "constant": ("constant", "constant"),
    "zero": ("constant", "constant"),
    "fill": ("constant", "constant"),
    "wrap": ("wrap", "wrap"),
    "circular": ("wrap", "wrap"),
}


def normalize_boundary(boundary: str) -> tuple[str, str]:
    """Devuelve el nombre canónico y el modo equivalente de SciPy.

    Parameters
    ----------
    boundary : str
        ``symmetric``, ``edge``, ``constant`` o ``wrap``, incluidos sus alias.

    Returns
    -------
    tuple of str
        Nombre público canónico y modo de ``scipy.ndimage``.

    Notes
    -----
    El análisis ya tiene geometría ``same``. ``valid`` no se implementa porque
    cambiaría la retícula espacial y exigiría otra síntesis.
    """
    if not isinstance(boundary, str):
        raise ValueError("boundary debe ser 'symmetric', 'edge', 'constant' o 'wrap'.")
    lowered = boundary.lower()
    if lowered == "same":
        raise ValueError("'same' ya es la geometría usada por el análisis.")
    if lowered == "valid":
        raise ValueError("'valid' cambia la retícula y no está implementado.")
    try:
        return _BOUNDARIES[lowered]
    except KeyError as exc:
        raise ValueError(
            "boundary debe ser 'symmetric', 'edge', 'constant' o 'wrap'."
        ) from exc


def cartesian_transform(
    image: Array,
    bank: Mapping,
    orders: Sequence[Order] | None = None,
    sampling_step: int = 1,
    boundary: str = "symmetric",
) -> dict[Order, Array]:
    """Calcula los coeficientes cartesianos ``L_mn``.

    Parameters
    ----------
    image : ndarray, shape (rows, columns)
        Imagen real bidimensional.
    bank : mapping
        Banco devuelto por ``build_filter_bank``.
    orders : sequence of tuple of int, optional
        Canales a calcular; por defecto se usan los órdenes públicos.
    sampling_step : int, default=1
        Separación de la retícula de análisis en píxeles.
    boundary : {"symmetric", "edge", "constant", "wrap"}
        Extensión del borde. ``edge`` repite el píxel más cercano.

    Returns
    -------
    dict
        Mapas 2-D indexados por ``(m, n)``. ``m`` filtra columnas y ``n`` filas.
    """
    if isinstance(sampling_step, (bool, np.bool_)) or not isinstance(
        sampling_step, (int, np.integer)
    ) or sampling_step < 1:
        raise ValueError("sampling_step debe ser un entero positivo.")
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("image debe ser una matriz 2-D finita no vacía.")

    selected_orders = list(bank["orders"] if orders is None else orders)
    missing = [
        order
        for order in selected_orders
        if order[0] not in bank["filters"] or order[1] not in bank["filters"]
    ]
    if missing:
        raise KeyError(f"El banco no contiene filtros para {missing[:5]}.")

    _, scipy_mode = normalize_boundary(boundary)
    horizontal = {
        m: correlate1d(
            values,
            bank["filters"][m],
            axis=1,
            mode=scipy_mode,
            cval=0.0,
        )
        for m in sorted({m for m, _ in selected_orders})
    }
    coefficients: dict[Order, Array] = {}
    for m, n in selected_orders:
        dense = correlate1d(
            horizontal[m],
            bank["filters"][n],
            axis=0,
            mode=scipy_mode,
            cval=0.0,
        )
        coefficients[(m, n)] = dense[::sampling_step, ::sampling_step].copy()
    return coefficients
