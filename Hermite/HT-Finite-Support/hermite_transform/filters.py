"""Órdenes y filtros Hermite--Gaussianos de soporte finito."""

import warnings

import numpy as np
from scipy.special import eval_hermite, gammaln


Array = np.ndarray
Order = tuple[int, int]
SUPPORT_TOLERANCE = 1e-8
TAIL_SAMPLES = 3
DC_WARNING_THRESHOLD = 1e-3


def _nonnegative_integer(name: str, value: int) -> int:
    """Devuelve ``value`` como entero no negativo o genera un error claro."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} debe ser un entero no negativo.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} debe ser un entero no negativo.")
    return value


def hermite_orders(max_order: int, region: str = "triangle") -> list[Order]:
    """Genera los pares ``(m, n)`` de la región solicitada.

    Parameters
    ----------
    max_order : int
        Orden total máximo para ``triangle`` u orden máximo por eje para
        ``square``.
    region : {"triangle", "square"}, default="triangle"
        Forma de la región de coeficientes.

    Returns
    -------
    list of tuple of int
        Órdenes agrupados por grado total: L00, L10, L01, L20, etc.
    """
    limit = _nonnegative_integer("max_order", max_order)
    if not isinstance(region, str) or region.lower() not in {"triangle", "square"}:
        raise ValueError("region debe ser 'triangle' o 'square'.")
    region = region.lower()

    orders: list[Order] = []
    maximum_total = limit if region == "triangle" else 2 * limit
    for total in range(maximum_total + 1):
        largest_m = total if region == "triangle" else min(limit, total)
        smallest_m = 0 if region == "triangle" else max(0, total - limit)
        orders.extend(
            (m, total - m)
            for m in range(largest_m, smallest_m - 1, -1)
        )
    return orders


def hermite_filter_1d(order: int, coordinates: Array, sigma: float) -> Array:
    """Evalúa una función de análisis Hermite--Gaussiana 1-D.

    Parameters
    ----------
    order : int
        Orden del polinomio de Hermite de físicos.
    coordinates : ndarray
        Coordenadas espaciales donde se muestrea el filtro.
    sigma : float
        Escala Gaussiana positiva en píxeles.

    Returns
    -------
    ndarray
        Valores de ``H_n(x/sigma) exp(-(x/sigma)^2)`` con normalización
        ``1 / (sigma sqrt(pi) sqrt(2^n n!))``.
    """
    order = _nonnegative_integer("order", order)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma debe ser un número finito positivo.")
    scaled = np.asarray(coordinates, dtype=np.float64) / float(sigma)
    normalization = np.exp(
        -0.5 * (order * np.log(2.0) + gammaln(order + 1.0))
    )
    polynomial = normalization * eval_hermite(order, scaled)
    gaussian = np.exp(-(scaled**2)) / (float(sigma) * np.sqrt(np.pi))
    return np.asarray(polynomial * gaussian, dtype=np.float64)


def choose_support(
    sigma: float,
    highest_order: int,
    tolerance: float = SUPPORT_TOLERANCE,
    tail_samples: int = TAIL_SAMPLES,
) -> int:
    """Selecciona el radio de un soporte simétrico de colas pequeñas.

    Parameters
    ----------
    sigma : float
        Escala Gaussiana en píxeles.
    highest_order : int
        Mayor orden 1-D requerido.
    tolerance : float, default=1e-8
        Cota relativa respecto al máximo de cada filtro.
    tail_samples : int, default=3
        Muestras consecutivas que deben satisfacer la cota.

    Returns
    -------
    int
        Radio del soporte impar ``[-radius, radius]``.
    """
    highest_order = _nonnegative_integer("highest_order", highest_order)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma debe ser un número finito positivo.")
    if not 0 < tolerance < 1 or tail_samples < 1:
        raise ValueError("tolerance y tail_samples deben ser positivos.")

    scale = max(4.0, np.sqrt(2.0 * highest_order + 1.0) + 2.0)
    radius = max(1, int(np.ceil(float(sigma) * scale)))
    while True:
        coordinates = np.arange(radius + int(tail_samples), dtype=np.float64)
        tails_are_small = True
        for order in range(highest_order + 1):
            values = np.abs(hermite_filter_1d(order, coordinates, sigma))
            if not np.all(values[radius:] / np.max(values) < tolerance):
                tails_are_small = False
                break
        if tails_are_small:
            return radius
        radius += 1


def _filter_diagnostics(filters: dict[int, Array]) -> dict[int, dict[str, float]]:
    """Calcula fuga DC, paridad y magnitud de borde de cada filtro."""
    dc_reference = max(abs(float(np.sum(filters[0]))), np.finfo(float).eps)
    diagnostics: dict[int, dict[str, float]] = {}
    for order, values in filters.items():
        peak = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
        signed_sum = float(np.sum(values))
        diagnostics[order] = {
            "sum": signed_sum,
            "relative_dc_leakage": abs(signed_sum) / dc_reference,
            "parity_max_abs_error": float(
                np.max(np.abs(values - ((-1) ** order) * values[::-1]))
            ),
            "edge_relative_magnitude": float(
                max(abs(values[0]), abs(values[-1])) / peak
            ),
        }
    return diagnostics


def build_filter_bank(
    max_order: int = 3,
    sigma: float = 2.0,
    region: str = "square",
) -> dict:
    """Construye el banco separable y sus diagnósticos discretos.

    Parameters
    ----------
    max_order : int, default=3
        Límite total para ``triangle`` o límite por eje para ``square``.
    sigma : float, default=2.0
        Escala Gaussiana en píxeles.
    region : {"triangle", "square"}, default="square"
        Región visible de coeficientes.

    Returns
    -------
    dict
        Filtros 1-D, órdenes públicos y auxiliares, soporte, ventanas y
        diagnósticos de muestreo.

    Notes
    -----
    El modo cuadrado usa internamente el triángulo completo hasta grado
    ``2*max_order`` para que el steering no opere sobre bloques incompletos.
    """
    visible_orders = hermite_orders(max_order, region)
    region = region.lower()
    steering_orders = (
        hermite_orders(2 * int(max_order), "triangle")
        if region == "square"
        else visible_orders.copy()
    )
    highest_order = max(max(order) for order in steering_orders)
    radius = choose_support(sigma, highest_order)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    filters = {
        order: hermite_filter_1d(order, coordinates, sigma)
        for order in range(highest_order + 1)
    }
    diagnostics = _filter_diagnostics(filters)
    problematic = [
        order
        for order in range(1, highest_order + 1)
        if diagnostics[order]["relative_dc_leakage"] > DC_WARNING_THRESHOLD
    ]
    if problematic:
        warnings.warn(
            "Los filtros muestreados presentan fuga DC apreciable en los "
            f"órdenes {problematic}; considere aumentar sigma.",
            RuntimeWarning,
            stacklevel=2,
        )

    window_squared = np.outer(filters[0], filters[0])
    return {
        "orders": visible_orders,
        "steering_orders": steering_orders,
        "filters": filters,
        "coordinates": coordinates,
        "radius": radius,
        "kernel_size": 2 * radius + 1,
        "window": np.sqrt(window_squared),
        "window_squared": window_squared,
        "diagnostics": diagnostics,
        "tail_tolerance": SUPPORT_TOLERANCE,
        "tail_samples": TAIL_SAMPLES,
        "sigma": float(sigma),
        "max_order": int(max_order),
        "region": region,
    }
