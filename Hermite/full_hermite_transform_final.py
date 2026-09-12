"""Transformada de Hermite 2-D con filtros Hermite--Gaussianos.

Esta version reune el flujo matematico de la implementacion simple con las
comprobaciones numericas y las salidas utiles de la version refactorizada. El
flujo completo es

``imagen -> analisis cartesiano -> steering -> steering inverso -> sintesis``.

La funcion de analisis unidimensional de orden ``n`` es

.. math::

    a_n(x;\sigma) = \frac{H_n(x/\sigma)}{\sqrt{2^n n!}}
                     \frac{e^{-(x/\sigma)^2}}{\sigma\sqrt{\pi}},

donde ``H_n`` es el polinomio de Hermite de fisicos. Los filtros 2-D son
separables: el primer indice ``m`` actua en columnas (eje ``x``) y el segundo
``n`` en filas (eje ``y``).

La sintesis implementada es una reconstruccion truncada por overlap-add. Usa
los mismos filtros Hermite--Gaussianos del analisis en el numerador y divide
por la suma de las ventanas Gaussianas de orden cero desplazadas. Por tanto,
no debe interpretarse como una inversa exacta cuando solo se conserva una
region finita de coeficientes.

El analisis produce geometria ``same`` antes del submuestreo. ``valid`` no se
ofrece como opcion porque cambiaria la reticula espacial y exigiria otra
definicion de sintesis. Las extensiones disponibles son simetrica, constante,
repeticion del borde y circular.
"""

from __future__ import annotations

import csv
import warnings
from math import comb
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import convolve1d, correlate1d
from scipy.special import eval_hermite, gammaln


Array = np.ndarray
Order = tuple[int, int]
CoefficientMaps = dict[Order, Array]

_SUPPORT_TOLERANCE = 1e-8
_TAIL_SAMPLES = 3
_DC_LEAKAGE_WARNING_THRESHOLD = 1e-3
_RGB_LUMINANCE_WEIGHTS = np.asarray(
    [0.298936021293775, 0.587043074451121, 0.114020904255103],
    dtype=np.float64,
)


# -----------------------------------------------------------------------------
# Lectura de imagenes y validacion basica
# -----------------------------------------------------------------------------


def _nonnegative_integer(name: str, value: int) -> int:
    """Valida un parametro entero no negativo.

    Parameters
    ----------
    name : str
        Nombre usado en el mensaje de error.
    value : int
        Valor que se desea validar.

    Returns
    -------
    int
        Valor convertido al tipo entero nativo de Python.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} debe ser un entero no negativo.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} debe ser un entero no negativo.")
    return result


def _positive_integer(name: str, value: int) -> int:
    """Valida un parametro entero estrictamente positivo.

    Parameters
    ----------
    name : str
        Nombre usado en el mensaje de error.
    value : int
        Valor que se desea validar.

    Returns
    -------
    int
        Valor positivo convertido al tipo entero nativo de Python.
    """
    result = _nonnegative_integer(name, value)
    if result == 0:
        raise ValueError(f"{name} debe ser un entero positivo.")
    return result


def _array_to_grayscale(image: Array, source: str) -> Array:
    """Convierte un array de imagen numerico a escala de grises.

    Parameters
    ----------
    image : ndarray
        Imagen 2-D o array con 1, 2, 3 o 4 canales en el ultimo eje.
    source : str
        Descripcion de la entrada usada en mensajes de error.

    Returns
    -------
    ndarray
        Imagen 2-D ``float64`` sin reescalar sus intensidades.

    Notes
    -----
    En entradas LA se ignora alpha. En RGB/RGBA se usa luminancia y tambien se
    ignora alpha. La escala original, por ejemplo 0--255 o 0--65535, se
    conserva.
    """
    array = np.asarray(image)

    if np.iscomplexobj(array):
        raise ValueError(f"{source} contiene valores complejos.")
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"{source} debe contener datos numericos.")

    if array.ndim == 2:
        grayscale = array
    elif array.ndim == 3:
        channels = array.shape[-1]
        if channels in {1, 2}:
            grayscale = array[..., 0]
        elif channels in {3, 4}:
            rgb = array[..., :3].astype(np.float64, copy=False)
            grayscale = np.tensordot(
                rgb,
                _RGB_LUMINANCE_WEIGHTS,
                axes=([-1], [0]),
            )
        else:
            raise ValueError(
                f"{source} tiene {channels} canales; se esperaban entre 1 y 4."
            )
    else:
        raise ValueError(
            f"{source} debe tener forma (alto, ancho) o "
            "(alto, ancho, canales)."
        )

    result = np.asarray(grayscale, dtype=np.float64)
    if result.size == 0:
        raise ValueError(f"{source} esta vacia.")
    if not np.isfinite(result).all():
        raise ValueError(f"{source} contiene NaN o valores infinitos.")
    return result


def _pil_to_grayscale(image: Image.Image, source: str) -> Array:
    """Interpreta una imagen PIL y conserva su profundidad numerica.

    Parameters
    ----------
    image : PIL.Image.Image
        Imagen PIL abierta.
    source : str
        Descripcion de la entrada usada en mensajes de error.

    Returns
    -------
    ndarray
        Imagen 2-D ``float64`` con la orientacion EXIF aplicada.
    """
    oriented = ImageOps.exif_transpose(image)
    mode = oriented.mode
    direct_modes = {"1", "L", "LA", "I", "F", "RGB", "RGBA", "RGBX"}

    if mode in direct_modes or mode.startswith("I;16"):
        array = np.asarray(oriented)
    else:
        # Las paletas, CMYK y otros espacios codificados deben ser
        # interpretados por PIL antes de calcular la luminancia.
        array = np.asarray(oriented.convert("RGB"))

    return _array_to_grayscale(array, f"{source} (modo PIL {mode!r})")


def read_image(image: str | Path | Image.Image | Array) -> Array:
    """Lee una imagen como un array bidimensional en escala de grises.

    Parameters
    ----------
    image : str, pathlib.Path, PIL.Image.Image or ndarray
        Ruta, imagen PIL o array numerico. Se admiten imagenes grayscale,
        grayscale con alpha, RGB y RGBA.

    Returns
    -------
    ndarray
        Imagen 2-D ``float64`` en su escala numerica original.

    Notes
    -----
    Las rutas y objetos PIL respetan la orientacion EXIF. Los TIFF grayscale
    de 16 bits conservan sus intensidades y no se reducen a 8 bits.
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as pil_image:
            return _pil_to_grayscale(pil_image, f"archivo {str(image)!r}")
    if isinstance(image, Image.Image):
        return _pil_to_grayscale(image, "imagen PIL")
    return _array_to_grayscale(image, "array de entrada")


# -----------------------------------------------------------------------------
# Ordenes y banco de filtros Hermite--Gaussianos
# -----------------------------------------------------------------------------


def hermite_orders(max_order: int, region: str = "triangle") -> list[Order]:
    """Genera los pares de orden de los coeficientes Hermite 2-D.

    Parameters
    ----------
    max_order : int
        Orden total maximo para ``triangle`` u orden maximo por eje para
        ``square``.
    region : {"triangle", "square"}, default="triangle"
        Forma de la region de coeficientes.

    Returns
    -------
    list of tuple of int
        Pares ``(m, n)`` ordenados como ``L00, L10, L01, L20, ...``.
    """
    limit = _nonnegative_integer("max_order", max_order)
    if not isinstance(region, str):
        raise ValueError("region debe ser 'triangle' o 'square'.")
    normalized_region = region.lower()
    if normalized_region not in {"triangle", "square"}:
        raise ValueError("region debe ser 'triangle' o 'square'.")

    orders: list[Order] = []
    maximum_total = limit if normalized_region == "triangle" else 2 * limit
    for total in range(maximum_total + 1):
        largest_m = total if normalized_region == "triangle" else min(limit, total)
        smallest_m = (
            0 if normalized_region == "triangle" else max(0, total - limit)
        )
        orders.extend(
            (m, total - m)
            for m in range(largest_m, smallest_m - 1, -1)
        )
    return orders


def hermite_filter_1d(order: int, coordinates: Array, sigma: float) -> Array:
    """Evalua un filtro Hermite--Gaussiano unidimensional normalizado.

    Parameters
    ----------
    order : int
        Orden no negativo del polinomio de Hermite de fisicos.
    coordinates : ndarray
        Coordenadas espaciales, normalmente enteros centrados en cero.
    sigma : float
        Escala Gaussiana positiva expresada en pixeles.

    Returns
    -------
    ndarray
        Valores ``float64`` de ``a_order(x; sigma)`` en las coordenadas dadas.
    """
    n = _nonnegative_integer("order", order)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma debe ser un numero finito positivo.")

    x = np.asarray(coordinates, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("coordinates debe contener solo valores finitos.")

    scaled = x / float(sigma)
    normalization = np.exp(
        -0.5 * (n * np.log(2.0) + gammaln(n + 1.0))
    )
    polynomial = normalization * eval_hermite(n, scaled)
    gaussian = np.exp(-(scaled**2)) / (float(sigma) * np.sqrt(np.pi))
    return np.asarray(polynomial * gaussian, dtype=np.float64)


def choose_support(
    sigma: float,
    highest_order: int,
    tolerance: float = _SUPPORT_TOLERANCE,
    tail_samples: int = _TAIL_SAMPLES,
) -> int:
    """Selecciona un soporte finito impar para todos los filtros requeridos.

    Parameters
    ----------
    sigma : float
        Escala Gaussiana positiva en pixeles.
    highest_order : int
        Mayor orden unidimensional que debe caber en el soporte.
    tolerance : float, default=1e-8
        Magnitud relativa maxima aceptada en la cola de cada filtro.
    tail_samples : int, default=3
        Numero de muestras consecutivas usadas para comprobar cada cola.

    Returns
    -------
    int
        Radio ``r`` del soporte simetrico ``[-r, r]``.

    Notes
    -----
    La prueba controla el truncamiento de las colas, no la ortogonalidad de la
    base despues del muestreo. Los diagnosticos discretos se calculan al
    construir el banco.
    """
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma debe ser un numero finito positivo.")
    highest = _nonnegative_integer("highest_order", highest_order)
    if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance debe estar entre cero y uno.")
    samples = _positive_integer("tail_samples", tail_samples)

    initial_scale = max(4.0, np.sqrt(2.0 * highest + 1.0) + 2.0)
    radius = max(1, int(np.ceil(float(sigma) * initial_scale)))
    maximum_radius = max(
        radius + 1000,
        int(np.ceil(50.0 * float(sigma))) + highest,
    )

    while radius <= maximum_radius:
        coordinates = np.arange(radius + samples, dtype=np.float64)
        acceptable = True
        for order in range(highest + 1):
            values = np.abs(hermite_filter_1d(order, coordinates, sigma))
            peak = float(np.max(values))
            if peak == 0.0 or not np.all(values[radius:] / peak < tolerance):
                acceptable = False
                break
        if acceptable:
            return radius
        radius += 1

    raise RuntimeError(
        "No fue posible encontrar un soporte finito para el orden y sigma "
        "solicitados."
    )


def build_filter_bank(
    max_order: int = 3,
    sigma: float = 2.0,
    region: str = "square",
) -> dict:
    """Construye el banco separable de filtros Hermite--Gaussianos.

    Parameters
    ----------
    max_order : int, default=3
        Orden total maximo en ``triangle`` u orden maximo por eje en
        ``square``.
    sigma : float, default=2.0
        Escala Gaussiana positiva en pixeles.
    region : {"triangle", "square"}, default="square"
        Region visible de coeficientes.

    Returns
    -------
    dict
        Banco con ordenes visibles y auxiliares, filtros 1-D, soporte,
        ventanas y diagnosticos de discretizacion.

    Notes
    -----
    En region cuadrada, el steering necesita bloques completos hasta orden
    total ``2*max_order``. Esos ordenes auxiliares se analizan internamente,
    aunque las salidas publicas sigan siendo cuadradas.
    """
    limit = _nonnegative_integer("max_order", max_order)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma debe ser un numero finito positivo.")
    sigma = float(sigma)

    visible_orders = hermite_orders(limit, region)
    normalized_region = region.lower()
    steering_orders = (
        hermite_orders(2 * limit, "triangle")
        if normalized_region == "square"
        else visible_orders.copy()
    )
    highest_order = max(
        (max(m, n) for m, n in steering_orders),
        default=0,
    )
    radius = choose_support(sigma, highest_order)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    filters = {
        order: hermite_filter_1d(order, coordinates, sigma)
        for order in range(highest_order + 1)
    }

    dc_reference = max(
        abs(float(np.sum(filters[0]))),
        np.finfo(np.float64).eps,
    )
    diagnostics: dict[int, dict[str, float]] = {}
    for order, values in filters.items():
        peak = max(float(np.max(np.abs(values))), np.finfo(np.float64).tiny)
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

    problematic_orders = [
        order
        for order in range(1, highest_order + 1)
        if diagnostics[order]["relative_dc_leakage"]
        > _DC_LEAKAGE_WARNING_THRESHOLD
    ]
    if problematic_orders:
        warnings.warn(
            "Los filtros Hermite--Gaussianos muestreados presentan fuga DC "
            f"apreciable en los ordenes {problematic_orders}. Considere aumentar "
            "sigma respecto al espaciado de pixeles.",
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
        "tail_tolerance": _SUPPORT_TOLERANCE,
        "tail_samples": _TAIL_SAMPLES,
        "sigma": sigma,
        "max_order": limit,
        "region": normalized_region,
    }


# -----------------------------------------------------------------------------
# Analisis cartesiano
# -----------------------------------------------------------------------------


def _normalize_boundary(boundary: str) -> tuple[str, str]:
    """Normaliza el nombre publico de una extension de borde.

    Parameters
    ----------
    boundary : str
        Nombre o alias del modo de borde.

    Returns
    -------
    tuple of str
        Nombre canonico y modo equivalente de ``scipy.ndimage``.

    Notes
    -----
    ``same`` no es una extension de borde: todos los modos ya conservan el
    tamano antes del submuestreo. ``valid`` no se admite porque cambia la
    reticula y requiere una sintesis diferente.
    """
    if not isinstance(boundary, str):
        raise ValueError(
            "boundary debe ser 'symmetric', 'edge', 'constant' o 'wrap'."
        )

    aliases = {
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
    lowered = boundary.lower()
    if lowered in {"same", "valid"}:
        explanation = (
            "'same' ya es la geometria de salida usada por el analisis."
            if lowered == "same"
            else "'valid' cambia la reticula y no esta implementado en esta version."
        )
        raise ValueError(explanation)
    try:
        return aliases[lowered]
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
) -> CoefficientMaps:
    """Calcula los mapas cartesianos ``L_mn`` mediante correlacion separable.

    Parameters
    ----------
    image : ndarray, shape (rows, columns)
        Imagen real bidimensional en su escala numerica original.
    bank : mapping
        Banco devuelto por :func:`build_filter_bank`.
    orders : sequence of tuple of int, optional
        Ordenes que se calcularan. Por defecto se usan los ordenes visibles.
    sampling_step : int, default=1
        Separacion en pixeles de la reticula de analisis.
    boundary : {"symmetric", "edge", "constant", "wrap"}, default="symmetric"
        Extension usada durante la correlacion. ``edge`` repite el valor del
        pixel mas cercano y ``constant`` usa ceros.

    Returns
    -------
    dict
        Mapas 2-D ``float64`` indexados por ``(m, n)``. Antes del submuestreo
        la geometria es ``same``.
    """
    step = _positive_integer("sampling_step", sampling_step)
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("image debe ser un array 2-D no vacio.")
    if not np.isfinite(values).all():
        raise ValueError("image contiene NaN o valores infinitos.")

    selected_orders = list(bank["orders"] if orders is None else orders)
    if not selected_orders:
        raise ValueError("Se requiere al menos un orden para el analisis.")
    missing_filters = [
        order
        for order in selected_orders
        if order[0] not in bank["filters"] or order[1] not in bank["filters"]
    ]
    if missing_filters:
        raise KeyError(
            f"El banco no contiene filtros para {missing_filters[:5]}."
        )

    _, scipy_mode = _normalize_boundary(boundary)
    horizontal: dict[int, Array] = {}
    for m, _ in selected_orders:
        if m not in horizontal:
            horizontal[m] = correlate1d(
                values,
                np.asarray(bank["filters"][m], dtype=np.float64),
                axis=1,
                mode=scipy_mode,
                cval=0.0,
            )

    coefficients: CoefficientMaps = {}
    for m, n in selected_orders:
        dense_response = correlate1d(
            horizontal[m],
            np.asarray(bank["filters"][n], dtype=np.float64),
            axis=0,
            mode=scipy_mode,
            cval=0.0,
        )
        coefficients[(m, n)] = dense_response[::step, ::step].copy()
    return coefficients


# -----------------------------------------------------------------------------
# Orientacion y steering
# -----------------------------------------------------------------------------


def dominant_theta(coefficients: Mapping[Order, Array]) -> Array:
    """Estima la direccion local del gradiente a partir del primer orden.

    Parameters
    ----------
    coefficients : mapping
        Mapas que deben contener ``L_10`` y ``L_01``.

    Returns
    -------
    ndarray
        Angulo ``atan2(L_01, L_10)`` en radianes y en el intervalo habitual
        ``[-pi, pi]``.

    Notes
    -----
    El angulo representa la direccion del gradiente, no la direccion tangente
    a un borde. En puntos de gradiente nulo, ``atan2(0, 0)`` devuelve cero por
    convencion numerica.
    """
    if (1, 0) not in coefficients or (0, 1) not in coefficients:
        raise ValueError(
            "La orientacion dominante requiere los coeficientes (1, 0) y (0, 1)."
        )
    return np.arctan2(
        np.asarray(coefficients[(0, 1)], dtype=np.float64),
        np.asarray(coefficients[(1, 0)], dtype=np.float64),
    )


def _theta_array(theta: float | Array, spatial_shape: tuple[int, int]) -> Array:
    """Convierte un angulo escalar o mapa a la forma espacial requerida.

    Parameters
    ----------
    theta : float or ndarray
        Angulo en radianes.
    spatial_shape : tuple of int
        Forma ``(rows, columns)`` de los mapas de coeficientes.

    Returns
    -------
    ndarray
        Mapa ``float64`` de angulos con forma ``spatial_shape``.
    """
    angle = np.asarray(theta, dtype=np.float64)
    if not np.isfinite(angle).all():
        raise ValueError("theta debe contener solo valores finitos.")
    if angle.ndim == 0:
        return np.full(spatial_shape, float(angle), dtype=np.float64)
    try:
        return np.broadcast_to(angle, spatial_shape).astype(
            np.float64,
            copy=False,
        )
    except ValueError as exc:
        raise ValueError(
            f"theta tiene forma {angle.shape}; se esperaba un escalar o un "
            f"mapa compatible con {spatial_shape}."
        ) from exc


def rotate_block(block: Array, theta: float | Array) -> Array:
    """Rota un bloque completo de igual orden total.

    Parameters
    ----------
    block : ndarray, shape (..., degree + 1)
        Bloque ordenado como ``[L_r0, L_(r-1)1, ..., L_0r]``.
    theta : float or ndarray
        Angulo de steering en radianes, escalar o mapa espacial.

    Returns
    -------
    ndarray
        Bloque rotado con la misma forma y normalizacion Hermite.

    Notes
    -----
    Las raices de los coeficientes binomiales aparecen por la normalizacion
    ``1/sqrt(2^n n!)`` de la base Hermite.
    """
    values = np.asarray(block, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("block debe contener al menos un coeficiente.")
    if not np.isfinite(values).all():
        raise ValueError("block contiene NaN o valores infinitos.")

    degree = values.shape[-1] - 1
    if degree == 0:
        return values.copy()

    angle_map = _theta_array(theta, values.shape[:-1])
    cosine = np.cos(angle_map)
    sine = np.sin(angle_map)
    pascal = np.sqrt(
        np.asarray([comb(degree, index) for index in range(degree + 1)])
    )

    work = values.copy()
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
) -> CoefficientMaps:
    """Aplica steering a todos los bloques de orden total solicitado.

    Parameters
    ----------
    coefficients : mapping
        Mapas cartesianos indexados por ``(m, n)``.
    theta : float or ndarray
        Angulo de rotacion en radianes.
    orders : sequence of tuple of int
        Ordenes completos que deben rotarse.

    Returns
    -------
    dict
        Mapas rotados con las mismas claves de ``orders``.

    Raises
    ------
    ValueError
        Si falta algun miembro de un bloque de igual orden total. En region
        cuadrada se deben proporcionar los ordenes auxiliares del banco.
    """
    selected_orders = list(orders)
    if not selected_orders:
        raise ValueError("orders debe contener al menos un orden.")
    missing = [order for order in selected_orders if order not in coefficients]
    if missing:
        raise ValueError(f"Faltan mapas de coeficientes: {missing[:5]}.")

    arrays = {
        order: np.asarray(coefficients[order], dtype=np.float64)
        for order in selected_orders
    }
    first_shape = arrays[selected_orders[0]].shape
    if len(first_shape) != 2:
        raise ValueError("Los mapas de coeficientes deben ser bidimensionales.")
    if any(value.shape != first_shape for value in arrays.values()):
        raise ValueError("Todos los mapas deben tener la misma forma.")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Los mapas contienen NaN o valores infinitos.")

    angle_map = _theta_array(theta, first_shape)
    result = {order: value.copy() for order, value in arrays.items()}
    order_set = set(selected_orders)

    for total in sorted({m + n for m, n in selected_orders}):
        block_orders = [
            (m, total - m)
            for m in range(total, -1, -1)
        ]
        missing_block = [order for order in block_orders if order not in order_set]
        if missing_block:
            raise ValueError(
                f"El steering requiere el bloque completo de orden total {total}; "
                f"faltan {missing_block}."
            )

        block = np.stack([arrays[order] for order in block_orders], axis=-1)
        rotated = rotate_block(block, angle_map)
        for index, order in enumerate(block_orders):
            result[order] = rotated[..., index]

    return result


def inverse_rotate_coefficients(
    coefficients: Mapping[Order, Array],
    theta: float | Array,
    orders: Sequence[Order],
) -> CoefficientMaps:
    """Recupera coeficientes cartesianos aplicando steering con ``-theta``.

    Parameters
    ----------
    coefficients : mapping
        Mapas de coeficientes previamente rotados.
    theta : float or ndarray
        Angulo directo en radianes.
    orders : sequence of tuple of int
        Bloques completos que se deben recuperar.

    Returns
    -------
    dict
        Coeficientes cartesianos recuperados.
    """
    return rotate_coefficients(
        coefficients,
        -np.asarray(theta, dtype=np.float64),
        orders,
    )


# -----------------------------------------------------------------------------
# Sintesis truncada
# -----------------------------------------------------------------------------


def synthesize(
    coefficients: Mapping[Order, Array],
    bank: Mapping,
    image_shape: tuple[int, int],
    sampling_step: int = 1,
) -> Array:
    """Reconstruye una imagen por overlap-add y normalizacion local.

    Parameters
    ----------
    coefficients : mapping
        Coeficientes cartesianos de todos los ordenes visibles del banco.
    bank : mapping
        Banco usado durante el analisis.
    image_shape : tuple of int
        Forma ``(alto, ancho)`` de la imagen que se desea reconstruir.
    sampling_step : int, default=1
        Separacion de la reticula usada durante el analisis.

    Returns
    -------
    ndarray
        Reconstruccion truncada ``float64`` con forma ``image_shape``.

    Notes
    -----
    Cada mapa se expande sobre la reticula, se convoluciona con los mismos
    filtros Hermite--Gaussianos del analisis y se suma. El denominador es la
    suma de los productos 2-D de los filtros de orden cero desplazados, es
    decir, la ventana Gaussiana ``w^2`` de esta convencion.
    """
    step = _positive_integer("sampling_step", sampling_step)
    if (
        not isinstance(image_shape, tuple)
        or len(image_shape) != 2
        or any(
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            or size <= 0
            for size in image_shape
        )
    ):
        raise ValueError("image_shape debe contener dos enteros positivos.")

    height, width = int(image_shape[0]), int(image_shape[1])
    rows = np.arange(0, height, step, dtype=int)
    columns = np.arange(0, width, step, dtype=int)
    expected_shape = (len(rows), len(columns))
    orders = list(bank["orders"])
    missing = [order for order in orders if order not in coefficients]
    if missing:
        raise KeyError(f"Faltan coeficientes para la sintesis: {missing[:5]}.")

    numerator = np.zeros((height, width), dtype=np.float64)
    for m, n in orders:
        coefficient_map = np.asarray(coefficients[(m, n)], dtype=np.float64)
        if coefficient_map.shape != expected_shape:
            raise ValueError(
                f"L_{m}{n} tiene forma {coefficient_map.shape}; se esperaba "
                f"{expected_shape}."
            )
        if not np.isfinite(coefficient_map).all():
            raise ValueError(f"L_{m}{n} contiene NaN o valores infinitos.")

        upsampled = np.zeros((height, width), dtype=np.float64)
        upsampled[np.ix_(rows, columns)] = coefficient_map
        horizontal = convolve1d(
            upsampled,
            np.asarray(bank["filters"][m], dtype=np.float64),
            axis=1,
            mode="constant",
            cval=0.0,
        )
        numerator += convolve1d(
            horizontal,
            np.asarray(bank["filters"][n], dtype=np.float64),
            axis=0,
            mode="constant",
            cval=0.0,
        )

    sampling_mask = np.zeros((height, width), dtype=np.float64)
    sampling_mask[np.ix_(rows, columns)] = 1.0
    denominator = convolve1d(
        sampling_mask,
        np.asarray(bank["filters"][0], dtype=np.float64),
        axis=1,
        mode="constant",
        cval=0.0,
    )
    denominator = convolve1d(
        denominator,
        np.asarray(bank["filters"][0], dtype=np.float64),
        axis=0,
        mode="constant",
        cval=0.0,
    )

    if np.any(denominator <= 0.0):
        raise ValueError(
            "sampling_step deja pixeles sin cobertura para el soporte elegido."
        )
    return np.divide(numerator, denominator)


# -----------------------------------------------------------------------------
# Resumenes numericos
# -----------------------------------------------------------------------------


def coefficients_to_stack(
    coefficients: Mapping[Order, Array],
    orders: Sequence[Order] | None = None,
) -> Array:
    """Apila mapas de coeficientes en un unico array multicanal.

    Parameters
    ----------
    coefficients : mapping
        Mapas 2-D indexados por orden.
    orders : sequence of tuple of int, optional
        Orden de los canales. Por defecto conserva el orden del mapping.

    Returns
    -------
    ndarray
        Array ``(rows, columns, channels)`` en ``float64``.
    """
    selected_orders = list(coefficients if orders is None else orders)
    if not selected_orders:
        raise ValueError("Se requiere al menos un mapa de coeficientes.")
    missing = [order for order in selected_orders if order not in coefficients]
    if missing:
        raise KeyError(f"Faltan coeficientes: {missing[:5]}.")

    values = [
        np.asarray(coefficients[order], dtype=np.float64)
        for order in selected_orders
    ]
    if values[0].ndim != 2:
        raise ValueError("Los mapas de coeficientes deben ser 2-D.")
    if any(value.shape != values[0].shape for value in values):
        raise ValueError("Todos los mapas deben tener la misma forma.")
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError("Los mapas contienen NaN o valores infinitos.")
    return np.stack(values, axis=-1)


def coefficient_energy(
    coefficients: Mapping[Order, Array],
    include_dc: bool = False,
) -> Array:
    """Calcula la energia raiz-de-suma-de-cuadrados de los coeficientes.

    Parameters
    ----------
    coefficients : mapping
        Mapas de coeficientes Hermite.
    include_dc : bool, default=False
        Si es ``True``, incluye ``L_00`` en la energia.

    Returns
    -------
    ndarray
        Imagen 2-D no negativa en la escala de los coeficientes.
    """
    if not isinstance(include_dc, (bool, np.bool_)):
        raise ValueError("include_dc debe ser booleano.")
    stack_orders = [
        order
        for order in coefficients
        if include_dc or order != (0, 0)
    ]
    if not coefficients:
        raise ValueError("Se requiere al menos un mapa de coeficientes.")
    if not stack_orders:
        first = np.asarray(next(iter(coefficients.values())), dtype=np.float64)
        return np.zeros_like(first)

    stack = coefficients_to_stack(coefficients, stack_orders)
    return np.sqrt(np.sum(stack**2, axis=-1))


def coefficient_roundtrip_metrics(
    original: Mapping[Order, Array],
    recovered: Mapping[Order, Array],
) -> dict[str, float]:
    """Mide el error de steering seguido por steering inverso.

    Parameters
    ----------
    original : mapping
        Coeficientes cartesianos antes de rotar.
    recovered : mapping
        Coeficientes recuperados despues de aplicar ``-theta``.

    Returns
    -------
    dict of str to float
        MSE, RMSE, MAE y error absoluto maximo entre coeficientes.
    """
    orders = list(original)
    if not orders:
        raise ValueError("Se requiere al menos un mapa de coeficientes.")
    missing = [order for order in orders if order not in recovered]
    if missing:
        raise KeyError(f"Faltan coeficientes recuperados: {missing[:5]}.")

    differences = np.concatenate(
        [
            (
                np.asarray(original[order], dtype=np.float64)
                - np.asarray(recovered[order], dtype=np.float64)
            ).ravel()
            for order in orders
        ]
    )
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
    """Calcula metricas escalares de la reconstruccion.

    Parameters
    ----------
    original : ndarray
        Imagen de referencia.
    reconstructed : ndarray
        Reconstruccion con la misma forma.
    data_range : float, optional
        Rango dinamico positivo para PSNR. Por defecto usa ``max-min`` de la
        imagen original, con una alternativa segura para imagenes constantes.

    Returns
    -------
    dict of str to float
        MSE, RMSE, MAE, error absoluto maximo y PSNR.
    """
    source = np.asarray(original, dtype=np.float64)
    estimate = np.asarray(reconstructed, dtype=np.float64)
    if source.shape != estimate.shape or source.size == 0:
        raise ValueError(
            "original y reconstructed deben tener la misma forma no vacia."
        )
    if not np.isfinite(source).all() or not np.isfinite(estimate).all():
        raise ValueError("Las imagenes deben contener valores finitos.")

    difference = source - estimate
    absolute = np.abs(difference)
    mse = float(np.mean(difference**2))
    rmse = float(np.sqrt(mse))

    if data_range is None:
        dynamic_range = float(np.max(source) - np.min(source))
        if dynamic_range == 0.0:
            dynamic_range = max(float(np.max(np.abs(source))), 1.0)
    else:
        dynamic_range = float(data_range)
        if not np.isfinite(dynamic_range) or dynamic_range <= 0.0:
            raise ValueError("data_range debe ser finito y positivo.")

    psnr = (
        float("inf")
        if mse == 0.0
        else float(20.0 * np.log10(dynamic_range / rmse))
    )
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": float(np.mean(absolute)),
        "max_abs_error": float(np.max(absolute)),
        "psnr": psnr,
    }


# -----------------------------------------------------------------------------
# Guardado de resultados
# -----------------------------------------------------------------------------


def _prepare_output_file(output_path: str | Path) -> Path:
    """Crea el directorio padre de un archivo de salida.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Ruta del archivo que se escribira.

    Returns
    -------
    pathlib.Path
        Ruta normalizada, lista para escritura.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_coefficients_grid(
    coefficients: Mapping[Order, Array],
    output_path: str | Path,
    title: str,
    cmap: str = "gray",
) -> None:
    """Guarda los mapas ``L_mn`` en una cuadricula por orden.

    Parameters
    ----------
    coefficients : mapping
        Mapas 2-D indexados por ``(m, n)``.
    output_path : str or pathlib.Path
        Archivo de imagen de destino.
    title : str
        Titulo general de la figura.
    cmap : str, default="gray"
        Colormap de Matplotlib.

    Returns
    -------
    None
        La funcion guarda y cierra la figura; no modifica los coeficientes.
    """
    if not coefficients:
        raise ValueError("Se requiere al menos un mapa de coeficientes.")
    orders = list(coefficients)
    values = {
        order: np.asarray(coefficients[order], dtype=np.float64)
        for order in orders
    }
    if any(value.ndim != 2 or value.size == 0 for value in values.values()):
        raise ValueError("Cada coeficiente debe ser un mapa 2-D no vacio.")
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("Los coeficientes contienen NaN o valores infinitos.")

    path = _prepare_output_file(output_path)
    max_m = max(m for m, _ in orders)
    max_n = max(n for _, n in orders)
    figure, axes = plt.subplots(
        max_n + 1,
        max_m + 1,
        figsize=(2.7 * (max_m + 1), 2.7 * (max_n + 1)),
        squeeze=False,
    )
    for n in range(max_n + 1):
        for m in range(max_m + 1):
            axis = axes[n, m]
            order = (m, n)
            if order in values:
                value = values[order]
                limit = max(
                    float(np.max(np.abs(value))),
                    np.finfo(np.float64).eps,
                )
                axis.imshow(value, cmap=cmap, vmin=-limit, vmax=limit)
                axis.set_title(rf"$L_{{{m},{n}}}$")
            axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_theta_image(theta: Array, output_path: str | Path) -> None:
    """Guarda un mapa angular en grados con un colormap ciclico.

    Parameters
    ----------
    theta : ndarray
        Mapa 2-D de angulos en radianes.
    output_path : str or pathlib.Path
        Archivo de imagen de destino.

    Returns
    -------
    None
        La funcion guarda y cierra la figura.
    """
    angle = np.asarray(theta, dtype=np.float64)
    if angle.ndim != 2 or angle.size == 0 or not np.isfinite(angle).all():
        raise ValueError("theta debe ser un mapa 2-D finito no vacio.")

    path = _prepare_output_file(output_path)
    figure, axis = plt.subplots(figsize=(7, 6))
    shown = axis.imshow(
        np.rad2deg(angle),
        cmap="twilight",
        vmin=-180.0,
        vmax=180.0,
    )
    axis.set_title(r"Direccion local del gradiente $\theta$ [grados]")
    axis.axis("off")
    figure.colorbar(shown, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_intensity_image(
    image: Array,
    output_path: str | Path,
    title: str,
) -> None:
    """Guarda una imagen escalar sin modificar su array numerico.

    Parameters
    ----------
    image : ndarray
        Imagen real 2-D.
    output_path : str or pathlib.Path
        Archivo de imagen de destino.
    title : str
        Titulo de la figura.

    Returns
    -------
    None
        La normalizacion de visualizacion solo afecta el archivo guardado.
    """
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("image debe ser una matriz 2-D finita no vacia.")

    path = _prepare_output_file(output_path)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if lower == upper:
        upper = lower + 1.0

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(values, cmap="gray", vmin=lower, vmax=upper)
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_reconstruction_comparison(
    original: Array,
    reconstructed: Array,
    output_path: str | Path,
) -> None:
    """Guarda la imagen original, la reconstruccion y su error absoluto.

    Parameters
    ----------
    original : ndarray
        Imagen original 2-D.
    reconstructed : ndarray
        Reconstruccion con la misma forma.
    output_path : str or pathlib.Path
        Archivo de imagen de destino.

    Returns
    -------
    None
        La funcion guarda y cierra una figura de tres paneles.
    """
    source = np.asarray(original, dtype=np.float64)
    estimate = np.asarray(reconstructed, dtype=np.float64)
    if source.shape != estimate.shape or source.ndim != 2 or source.size == 0:
        raise ValueError(
            "original y reconstructed deben ser matrices 2-D no vacias "
            "con la misma forma."
        )
    if not np.isfinite(source).all() or not np.isfinite(estimate).all():
        raise ValueError("Las imagenes deben contener valores finitos.")

    path = _prepare_output_file(output_path)
    error = np.abs(source - estimate)
    lower = float(min(np.min(source), np.min(estimate)))
    upper = float(max(np.max(source), np.max(estimate)))
    if lower == upper:
        upper = lower + 1.0

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(source, cmap="gray", vmin=lower, vmax=upper)
    axes[0].set_title("Original")
    axes[1].imshow(estimate, cmap="gray", vmin=lower, vmax=upper)
    axes[1].set_title("Reconstruccion truncada")
    shown_error = axes[2].imshow(error, cmap="gray", vmin=0.0)
    axes[2].set_title(r"Error absoluto $|I-\hat I|$")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(shown_error, ax=axes[2], fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_metrics_csv(
    metrics: Mapping[str, float],
    output_path: str | Path,
) -> None:
    """Guarda metricas escalares en un archivo CSV de dos columnas.

    Parameters
    ----------
    metrics : mapping of str to float
        Nombres y valores de las metricas.
    output_path : str or pathlib.Path
        Archivo CSV de destino.

    Returns
    -------
    None
        La funcion escribe encabezados ``metric`` y ``value``.
    """
    path = _prepare_output_file(output_path)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


# -----------------------------------------------------------------------------
# Flujo principal
# -----------------------------------------------------------------------------


def hermite_transform_image(
    image: str | Path | Image.Image | Array,
    max_order: int = 3,
    sigma: float = 2.0,
    coefficient_region: str = "square",
    sampling_step: int = 1,
    boundary: str = "symmetric",
    use_rotation: bool = True,
    use_inverse_rotation: bool = True,
    use_inverse_transform: bool = True,
    rotation_mode: str = "dominant",
    angle: float | Array = 0.0,
    angle_unit: str = "degrees",
    results_path: str | Path = "results",
) -> dict:
    """Ejecuta y guarda el flujo completo de la transformada de Hermite.

    Parameters
    ----------
    image : str, pathlib.Path, PIL.Image.Image or ndarray
        Imagen de entrada grayscale, RGB o RGBA.
    max_order : int, default=3
        Orden total maximo en ``triangle`` u orden maximo por eje en
        ``square``.
    sigma : float, default=2.0
        Escala Gaussiana positiva en pixeles.
    coefficient_region : {"triangle", "square"}, default="square"
        Region visible de coeficientes.
    sampling_step : int, default=1
        Separacion espacial de la reticula de analisis.
    boundary : {"symmetric", "edge", "constant", "wrap"}, default="symmetric"
        Extension del borde durante el analisis. ``edge`` repite el valor de
        la fila o columna mas cercana.
    use_rotation : bool, default=True
        Calcula theta y los coeficientes dirigidos.
    use_inverse_rotation : bool, default=True
        Recupera los coeficientes cartesianos despues del steering.
    use_inverse_transform : bool, default=True
        Calcula la sintesis truncada. Si hay steering, primero recupera los
        coeficientes cartesianos aunque ``use_inverse_rotation`` sea ``False``.
    rotation_mode : {"dominant", "fixed"}, default="dominant"
        ``dominant`` usa ``atan2(L_01, L_10)`` y ``fixed`` usa ``angle``.
    angle : float or ndarray, default=0.0
        Angulo fijo escalar o mapa compatible con los coefficient maps.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Unidad de ``angle`` en modo fijo.
    results_path : str or pathlib.Path, default="results"
        Carpeta unica donde se guardan todas las figuras y metricas. Se crea
        automaticamente, incluyendo sus directorios padre.

    Returns
    -------
    dict
        Imagen original, coeficientes cartesianos/rotados/recuperados, theta,
        reconstruccion, metricas, stack, energia, banco de filtros, ordenes,
        carpeta de resultados y mapping de archivos guardados.

    Notes
    -----
    Las salidas se guardan conforme cada etapa esta disponible. En region
    cuadrada se calculan internamente bloques triangulares completos hasta
    orden ``2*max_order`` para no truncar incorrectamente el steering.
    """
    flags = (use_rotation, use_inverse_rotation, use_inverse_transform)
    if not all(isinstance(flag, (bool, np.bool_)) for flag in flags):
        raise ValueError("Los controles del flujo deben ser booleanos.")
    if use_inverse_rotation and not use_rotation:
        raise ValueError(
            "use_inverse_rotation=True requiere use_rotation=True."
        )

    canonical_boundary, _ = _normalize_boundary(boundary)
    image_array = read_image(image)
    results_directory = Path(results_path)
    results_directory.mkdir(parents=True, exist_ok=True)
    saved_files: dict[str, Path] = {}

    original_path = results_directory / "01_original_input_grayscale_image.png"
    save_intensity_image(
        image_array,
        original_path,
        "Imagen original en escala de grises",
    )
    saved_files["original_image"] = original_path

    bank = build_filter_bank(
        max_order=max_order,
        sigma=sigma,
        region=coefficient_region,
    )
    visible_orders = list(bank["orders"])
    steering_orders = list(bank["steering_orders"])
    analysis_orders = steering_orders if use_rotation else visible_orders

    all_cartesian = cartesian_transform(
        image_array,
        bank,
        orders=analysis_orders,
        sampling_step=sampling_step,
        boundary=canonical_boundary,
    )
    cartesian = {
        order: all_cartesian[order].copy()
        for order in visible_orders
    }
    cartesian_path = (
        results_directory / "02_cartesian_hermite_coefficient_maps.png"
    )
    save_coefficients_grid(
        cartesian,
        cartesian_path,
        "Coeficientes Hermite cartesianos",
    )
    saved_files["cartesian_coefficients"] = cartesian_path

    theta: float | Array | None = None
    rotated: CoefficientMaps | None = None
    recovered: CoefficientMaps | None = None
    coefficient_metrics: dict[str, float] | None = None

    if use_rotation:
        if not isinstance(rotation_mode, str):
            raise ValueError("rotation_mode debe ser 'dominant' o 'fixed'.")
        mode = rotation_mode.lower()
        if mode in {"dominant", "gradient", "grad"}:
            theta = dominant_theta(all_cartesian)
        elif mode in {"fixed", "angle"}:
            if not isinstance(angle_unit, str):
                raise ValueError("angle_unit debe ser 'degrees' o 'radians'.")
            unit = angle_unit.lower()
            if unit in {"degree", "degrees", "deg"}:
                theta = np.deg2rad(np.asarray(angle, dtype=np.float64))
            elif unit in {"radian", "radians", "rad"}:
                theta = np.asarray(angle, dtype=np.float64)
            else:
                raise ValueError("angle_unit debe ser 'degrees' o 'radians'.")
        else:
            raise ValueError("rotation_mode debe ser 'dominant' o 'fixed'.")

        all_rotated = rotate_coefficients(
            all_cartesian,
            theta,
            steering_orders,
        )
        rotated = {
            order: all_rotated[order].copy()
            for order in visible_orders
        }
        rotated_path = (
            results_directory
            / "03_steered_rotated_hermite_coefficient_maps.png"
        )
        save_coefficients_grid(
            rotated,
            rotated_path,
            "Coeficientes Hermite rotados por steering",
        )
        saved_files["rotated_coefficients"] = rotated_path

        theta_path = (
            results_directory
            / "04_local_gradient_orientation_theta_in_degrees.png"
        )
        theta_map = _theta_array(theta, next(iter(cartesian.values())).shape)
        save_theta_image(theta_map, theta_path)
        saved_files["theta"] = theta_path

        if use_inverse_rotation or use_inverse_transform:
            all_recovered = inverse_rotate_coefficients(
                all_rotated,
                theta,
                steering_orders,
            )
            recovered = {
                order: all_recovered[order].copy()
                for order in visible_orders
            }
            coefficient_metrics = coefficient_roundtrip_metrics(
                cartesian,
                recovered,
            )
            recovered_path = (
                results_directory
                / "05_recovered_cartesian_from_rotated_coefficients.png"
            )
            save_coefficients_grid(
                recovered,
                recovered_path,
                "Coeficientes cartesianos recuperados desde los rotados",
            )
            saved_files["recovered_cartesian_coefficients"] = recovered_path

    reconstructed: Array | None = None
    image_metrics: dict[str, float] | None = None
    if use_inverse_transform:
        synthesis_coefficients = (
            recovered if use_rotation else cartesian
        )
        if synthesis_coefficients is None:
            raise RuntimeError(
                "No hay coeficientes cartesianos disponibles para la sintesis."
            )
        reconstructed = synthesize(
            synthesis_coefficients,
            bank,
            image_shape=image_array.shape,
            sampling_step=sampling_step,
        )
        image_metrics = reconstruction_metrics(image_array, reconstructed)

        reconstructed_path = (
            results_directory
            / "06_truncated_hermite_synthesis_reconstructed_image.png"
        )
        save_intensity_image(
            reconstructed,
            reconstructed_path,
            "Imagen reconstruida por sintesis Hermite truncada",
        )
        saved_files["reconstructed_image"] = reconstructed_path

        comparison_path = (
            results_directory
            / "07_reconstruction_comparison_with_absolute_error.png"
        )
        save_reconstruction_comparison(
            image_array,
            reconstructed,
            comparison_path,
        )
        saved_files["reconstruction_comparison"] = comparison_path

    active_coefficients = rotated if use_rotation else cartesian
    if active_coefficients is None:
        raise RuntimeError("No se generaron coeficientes activos.")
    coefficient_stack = coefficients_to_stack(
        active_coefficients,
        visible_orders,
    )
    energy = coefficient_energy(active_coefficients, include_dc=False)
    energy_kind = "rotated" if use_rotation else "cartesian"
    energy_path = (
        results_directory
        / f"08_{energy_kind}_hermite_coefficient_energy_without_dc.png"
    )
    save_intensity_image(
        energy,
        energy_path,
        "Energia de coeficientes Hermite sin componente DC",
    )
    saved_files["coefficient_energy"] = energy_path

    all_metrics: dict[str, float] = {}
    if coefficient_metrics is not None:
        all_metrics.update(coefficient_metrics)
    if image_metrics is not None:
        all_metrics.update(image_metrics)
    if all_metrics:
        metrics_path = (
            results_directory
            / "09_steering_roundtrip_and_reconstruction_metrics.csv"
        )
        save_metrics_csv(all_metrics, metrics_path)
        saved_files["metrics_csv"] = metrics_path

    return {
        "original_image": image_array,
        "cartesian_coefficients": cartesian,
        "rotated_coefficients": rotated,
        "recovered_cartesian_coefficients": recovered,
        "theta": theta,
        "reconstructed_image": reconstructed,
        "coefficient_roundtrip_metrics": coefficient_metrics,
        "reconstruction_metrics": image_metrics,
        "active_coefficients": active_coefficients,
        "coeff_stack": coefficient_stack,
        "coefficient_energy": energy,
        "transformed_image": energy,
        "orders": visible_orders,
        "auxiliary_orders": steering_orders,
        "filter_bank": bank,
        "support_radius": bank["radius"],
        "boundary": canonical_boundary,
        "results_path": results_directory,
        "saved_files": saved_files,
    }


if __name__ == "__main__":
    demo_image = Path("Fusion_Images_ds") / "house.tif"
    # demo_image = Path("Fusion_Images_ds") / "lena.jpg"
    demo_results = hermite_transform_image(
        image=demo_image,
        max_order=3,
        sigma=2.0,
        coefficient_region="square",
        sampling_step=1,
        boundary="symmetric",
        use_rotation=True,
        use_inverse_rotation=True,
        use_inverse_transform=True,
        rotation_mode="dominant",
        results_path=Path(__file__).resolve().parent / "results",
    )
    print("Ordenes visibles:", demo_results["orders"])
    print("Radio del soporte:", demo_results["support_radius"])
    print("Metricas de steering:", demo_results["coefficient_roundtrip_metrics"])
    print("Metricas de reconstruccion:", demo_results["reconstruction_metrics"])
    print("Resultados guardados en:", demo_results["results_path"])
