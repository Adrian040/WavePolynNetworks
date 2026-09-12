"""Métricas y representaciones multicanal de la transformada."""

from collections.abc import Mapping, Sequence

import numpy as np


Array = np.ndarray
Order = tuple[int, int]


def coefficients_to_stack(
    coefficients: Mapping[Order, Array],
    orders: Sequence[Order] | None = None,
) -> Array:
    """Apila coefficient maps en el último eje.

    Parameters
    ----------
    coefficients : mapping
        Mapas 2-D indexados por orden.
    orders : sequence of tuple of int, optional
        Orden de los canales; por defecto conserva el orden del mapping.

    Returns
    -------
    ndarray
        Array ``(rows, columns, channels)`` en ``float64``.
    """
    orders = list(coefficients if orders is None else orders)
    if not orders:
        raise ValueError("Se requiere al menos un mapa de coeficientes.")
    values = [np.asarray(coefficients[order], dtype=np.float64) for order in orders]
    if any(value.shape != values[0].shape for value in values):
        raise ValueError("Todos los mapas deben tener la misma forma.")
    return np.stack(values, axis=-1)


def coefficient_energy(
    coefficients: Mapping[Order, Array],
    include_dc: bool = False,
) -> Array:
    """Calcula la energía raíz-de-suma-de-cuadrados por píxel.

    Parameters
    ----------
    coefficients : mapping
        Mapas de coeficientes Hermite.
    include_dc : bool, default=False
        Incluye ``L_00`` cuando es verdadero.

    Returns
    -------
    ndarray
        Imagen de energía no negativa.
    """
    orders = [order for order in coefficients if include_dc or order != (0, 0)]
    if not orders:
        return np.zeros_like(next(iter(coefficients.values())), dtype=np.float64)
    stack = coefficients_to_stack(coefficients, orders)
    return np.sqrt(np.sum(stack**2, axis=-1))


def coefficient_roundtrip_metrics(
    original: Mapping[Order, Array],
    recovered: Mapping[Order, Array],
) -> dict[str, float]:
    """Mide el error del ciclo steering--steering inverso.

    Parameters
    ----------
    original : mapping
        Coeficientes cartesianos de referencia.
    recovered : mapping
        Coeficientes cartesianos recuperados.

    Returns
    -------
    dict of str to float
        MSE, RMSE, MAE y error absoluto máximo.
    """
    differences = np.concatenate(
        [
            (np.asarray(original[order]) - np.asarray(recovered[order])).ravel()
            for order in original
        ]
    ).astype(np.float64)
    mse = float(np.mean(differences**2))
    return {
        "coefficient_mse": mse,
        "coefficient_rmse": float(np.sqrt(mse)),
        "coefficient_mae": float(np.mean(np.abs(differences))),
        "coefficient_max_abs_error": float(np.max(np.abs(differences))),
    }


def reconstruction_metrics(
    original: Array,
    reconstructed: Array,
    data_range: float | None = None,
) -> dict[str, float]:
    """Calcula métricas de error y PSNR de la reconstrucción.

    Parameters
    ----------
    original, reconstructed : ndarray
        Imágenes con la misma forma.
    data_range : float, optional
        Rango dinámico usado por PSNR; por defecto se obtiene de ``original``.

    Returns
    -------
    dict of str to float
        MSE, RMSE, MAE, error máximo y PSNR.
    """
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    if original.shape != reconstructed.shape:
        raise ValueError("Las imágenes deben tener la misma forma.")
    difference = original - reconstructed
    mse = float(np.mean(difference**2))
    rmse = float(np.sqrt(mse))

    if data_range is None:
        data_range = float(np.max(original) - np.min(original))
        if data_range == 0:
            data_range = max(float(np.max(np.abs(original))), 1.0)
    elif not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range debe ser finito y positivo.")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": float(np.mean(np.abs(difference))),
        "max_abs_error": float(np.max(np.abs(difference))),
        "psnr": (
            float("inf")
            if mse == 0
            else float(20.0 * np.log10(float(data_range) / rmse))
        ),
    }
