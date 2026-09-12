"""Utilidades de visualización, identificación y comparación para la DHT.

Este módulo no contiene lógica de análisis, steering ni síntesis. Sus funciones
públicas permiten localizar canales, graficar coefficient maps y calcular
métricas básicas de reconstrucción o comparación.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _orders_list(orders: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    try:
        result = [tuple(order) for order in orders]
    except TypeError as exc:
        raise ValueError("orders debe ser una secuencia de pares (m, n).") from exc
    if any(
        len(order) != 2
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for value in order
        )
        for order in result
    ):
        raise ValueError("orders debe ser una secuencia de pares enteros no negativos.")
    normalized = [(int(m), int(n)) for m, n in result]
    if len(set(normalized)) != len(normalized):
        raise ValueError("orders no debe contener pares repetidos.")
    return normalized


def _coefficient_stack(
    coefficients: np.ndarray,
    orders: Sequence[Sequence[int]],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    normalized_orders = _orders_list(orders)
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("coefficients debe ser un stack 3-D (rows, columns, channels).")
    if values.shape[-1] != len(normalized_orders):
        raise ValueError(
            f"orders contiene {len(normalized_orders)} pares, pero el stack tiene "
            f"{values.shape[-1]} canales."
        )
    return values, normalized_orders


def coefficient_index(
    orders: Sequence[Sequence[int]],
    order: Sequence[int],
) -> int:
    """Obtiene el índice de canal asociado a un componente ``L_{m,n}``.

    Parameters
    ----------
    orders : sequence of pairs
        Orden de canales, normalmente producido por :func:`dhts_py.dhtord`.
    order : sequence of two int
        Par ``(m, n)`` que se desea localizar.

    Returns
    -------
    index : int
        Posición del coefficient map en el último eje del stack.
    """

    normalized_orders = _orders_list(orders)
    try:
        requested = tuple(order)
    except TypeError as exc:
        raise ValueError("order debe ser un par (m, n).") from exc
    if len(requested) != 2:
        raise ValueError("order debe ser un par (m, n).")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        for value in requested
    ):
        raise ValueError("order debe ser un par de enteros no negativos.")
    requested = (int(requested[0]), int(requested[1]))
    try:
        return normalized_orders.index(requested)
    except ValueError as exc:
        raise ValueError(f"El componente L_{{{requested[0]},{requested[1]}}} no está disponible.") from exc


def _display_stack(values: np.ndarray, normalize: str | None) -> tuple[np.ndarray, float]:
    if normalize not in {"individual", "global", None}:
        raise ValueError('normalize debe ser "individual", "global" o None.')

    if normalize == "individual":
        scales = np.max(np.abs(values), axis=(0, 1), keepdims=True)
        scales = np.where(scales == 0.0, 1.0, scales)
        return values / scales, 1.0

    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        scale = 1.0
    if normalize == "global":
        return values / scale, 1.0
    return values, scale


def plot_coefficients(
    coefficients: np.ndarray,
    orders: Sequence[Sequence[int]],
    coefficient_region: str = "triangle",
    normalize: str | None = "individual",
    *,
    cmap: str = "coolwarm",
    figsize: tuple[float, float] | None = None,
):
    """Muestra los coefficient maps en su posición conceptual.

    Parameters
    ----------
    coefficients : ndarray, shape (rows, columns, C)
        Stack de coefficient maps.
    orders : sequence of pairs
        Pares ``(m, n)`` asociados a los canales.
    coefficient_region : {"triangle", "square"}, default="triangle"
        Layout triangular por orden total o cuadrícula con filas ``n`` y
        columnas ``m``.
    normalize : {"individual", "global", None}, default="individual"
        Escala usada sólo para mostrar copias de los mapas. Nunca modifica la
        entrada.
    cmap : str, default="coolwarm"
        Colormap de Matplotlib; conviene que sea divergente.
    figsize : tuple, optional
        Tamaño explícito de la figura.

    Returns
    -------
    figure : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes
        Ejes organizados según la región solicitada.
    """

    values, normalized_orders = _coefficient_stack(coefficients, orders)
    if coefficient_region not in {"triangle", "square"}:
        raise ValueError('coefficient_region debe ser "triangle" o "square".')
    shown, limit = _display_stack(values, normalize)

    if coefficient_region == "triangle":
        max_total = max(sum(order) for order in normalized_orders)
        shape = (max_total + 1, 2 * max_total + 1)
        if figsize is None:
            figsize = (2.1 * shape[1], 2.1 * shape[0])
        figure, axes = plt.subplots(*shape, figsize=figsize, squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for channel, (m, n) in enumerate(normalized_orders):
            total = m + n
            column = max_total - total + 2 * n
            axis = axes[total, column]
            axis.imshow(shown[..., channel], cmap=cmap, vmin=-limit, vmax=limit)
            axis.set_title(rf"$L_{{{m},{n}}}$")
            axis.axis("off")
    else:
        max_m = max(m for m, _ in normalized_orders)
        max_n = max(n for _, n in normalized_orders)
        shape = (max_n + 1, max_m + 1)
        if figsize is None:
            figsize = (3.0 * shape[1], 3.0 * shape[0])
        figure, axes = plt.subplots(*shape, figsize=figsize, squeeze=False)
        index = {order: channel for channel, order in enumerate(normalized_orders)}
        for n in range(shape[0]):
            for m in range(shape[1]):
                axis = axes[n, m]
                axis.axis("off")
                if (m, n) in index:
                    axis.imshow(shown[..., index[(m, n)]], cmap=cmap, vmin=-limit, vmax=limit)
                    axis.set_title(rf"$L_{{{m},{n}}}$")

    figure.tight_layout()
    return figure, axes


def reconstruction_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    data_range: float | None = None,
) -> dict[str, float]:
    """Calcula métricas básicas entre una imagen y su reconstrucción.

    Parameters
    ----------
    original, reconstructed : ndarray
        Arrays numéricos con el mismo shape.
    data_range : float, optional
        Rango dinámico usado por PSNR. Por defecto se usa el rango del dtype
        entero o ``max(original) - min(original)`` para datos de punto flotante.

    Returns
    -------
    metrics : dict
        Valores ``mse``, ``rmse``, ``mae``, ``max_abs_error`` y ``psnr``.
    """

    original_array = np.asarray(original)
    reconstructed_array = np.asarray(reconstructed)
    if original_array.shape != reconstructed_array.shape:
        raise ValueError("original y reconstructed deben tener el mismo shape.")
    if original_array.size == 0:
        raise ValueError("Las imágenes no pueden estar vacías.")

    difference = original_array.astype(np.float64) - reconstructed_array.astype(np.float64)
    absolute = np.abs(difference)
    mse = float(np.mean(difference**2))
    rmse = float(np.sqrt(mse))

    if data_range is None:
        if np.issubdtype(original_array.dtype, np.integer):
            limits = np.iinfo(original_array.dtype)
            dynamic_range = float(limits.max - limits.min)
        else:
            source = original_array.astype(np.float64, copy=False)
            dynamic_range = float(np.max(source) - np.min(source))
            if dynamic_range == 0.0:
                dynamic_range = max(float(np.max(np.abs(source))), 1.0)
    else:
        dynamic_range = float(data_range)
        if not np.isfinite(dynamic_range) or dynamic_range <= 0.0:
            raise ValueError("data_range debe ser un número finito mayor que cero.")

    psnr = float("inf") if mse == 0.0 else float(20.0 * np.log10(dynamic_range / rmse))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": float(np.mean(absolute)),
        "max_abs_error": float(np.max(absolute)),
        "psnr": psnr,
    }


def plot_reconstruction_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    *,
    data_range: float | None = None,
    figsize: tuple[float, float] = (12.0, 4.0),
):
    """Muestra original, reconstrucción y error absoluto.

    Parameters
    ----------
    original, reconstructed : ndarray, shape (rows, columns)
        Imágenes grayscale con el mismo shape.
    data_range : float, optional
        Rango dinámico usado para PSNR.
    figsize : tuple, default=(12, 4)
        Tamaño de la figura.

    Returns
    -------
    figure : matplotlib.figure.Figure
    axes : ndarray, shape (3,)
    """

    original_array = np.asarray(original)
    reconstructed_array = np.asarray(reconstructed)
    if original_array.ndim != 2 or reconstructed_array.ndim != 2:
        raise ValueError("original y reconstructed deben ser imágenes grayscale 2-D.")
    metrics = reconstruction_metrics(original_array, reconstructed_array, data_range=data_range)
    error = np.abs(original_array.astype(np.float64) - reconstructed_array.astype(np.float64))
    lower = float(min(np.min(original_array), np.min(reconstructed_array)))
    upper = float(max(np.max(original_array), np.max(reconstructed_array)))

    figure, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(original_array, cmap="gray", vmin=lower, vmax=upper)
    axes[0].set_title("Original")
    axes[1].imshow(reconstructed_array, cmap="gray", vmin=lower, vmax=upper)
    axes[1].set_title("Reconstrucción")
    axes[2].imshow(error, cmap="magma", vmin=0.0)
    axes[2].set_title("Error absoluto")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        f"MSE={metrics['mse']:.3e} · RMSE={metrics['rmse']:.3e} · "
        f"MAE={metrics['mae']:.3e} · PSNR={metrics['psnr']:.2f} dB"
    )
    figure.tight_layout()
    return figure, axes


def compare_coefficients(
    first: np.ndarray,
    second: np.ndarray,
    orders: Sequence[Sequence[int]],
) -> dict[tuple[int, int], dict[str, float]]:
    """Compara dos stacks canal por canal.

    Parameters
    ----------
    first, second : ndarray, shape (rows, columns, C)
        Stacks con el mismo shape y orden de canales.
    orders : sequence of pairs
        Correspondencia de cada canal con ``L_{m,n}``.

    Returns
    -------
    comparison : dict
        Diccionario indexado por ``(m, n)``. Cada entrada contiene ``mse``,
        ``rmse``, ``mae`` y ``max_abs_error``.
    """

    first_array, normalized_orders = _coefficient_stack(first, orders)
    second_array, _ = _coefficient_stack(second, normalized_orders)
    if first_array.shape != second_array.shape:
        raise ValueError("first y second deben tener el mismo shape.")

    result: dict[tuple[int, int], dict[str, float]] = {}
    for channel, order in enumerate(normalized_orders):
        difference = first_array[..., channel] - second_array[..., channel]
        absolute = np.abs(difference)
        mse = float(np.mean(difference**2))
        result[order] = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(absolute)),
            "max_abs_error": float(np.max(absolute)),
        }
    return result


__all__ = [
    "coefficient_index",
    "plot_coefficients",
    "reconstruction_metrics",
    "plot_reconstruction_comparison",
    "compare_coefficients",
]
