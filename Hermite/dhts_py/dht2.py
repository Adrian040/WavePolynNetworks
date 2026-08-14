"""Transformada Discreta de Hermite bidimensional cartesiana.

Este módulo calcula los coefficient maps ``L_{m,n}`` mediante filtrado
separable con el banco discreto de Hermite y sampling espacial de paso ``T``.
La función pública es :func:`dht2`.
"""

import numpy as np
from scipy.signal import convolve2d

from .dhtmtx import dhtmtx
from .dhtord import dhtord


def _sampling_step(T: int) -> int:
    if isinstance(T, (bool, np.bool_)) or not isinstance(T, (int, np.integer)) or int(T) < 1:
        raise ValueError("T debe ser un entero >= 1.")
    return int(T)


def _symmetric_indices(length: int, width: int) -> np.ndarray:
    """Índices de una extensión que refleja incluyendo la muestra de borde."""

    positions = np.arange(-width, length + width, dtype=int)
    period = 2 * length
    folded = np.mod(positions, period)
    return np.where(folded < length, folded, period - 1 - folded)


def _symmetric_extension(image: np.ndarray, N: int) -> np.ndarray:
    width = (N + 1) // 2
    if width == 0:
        return image
    rows = _symmetric_indices(image.shape[0], width)
    columns = _symmetric_indices(image.shape[1], width)
    return image[np.ix_(rows, columns)]


def dht2(
    image: np.ndarray,
    N: int,
    D: int,
    T: int = 1,
    coefficient_region: str = "triangle",
    shape: str = "full",
) -> np.ndarray:
    """Calcula la DHT cartesiana de una imagen grayscale.

    Parameters
    ----------
    image : ndarray, shape (rows, columns)
        Imagen grayscale bidimensional. Sus valores se usan sin normalización
        ni clipping y los cálculos se realizan en ``float64``.
    N : int
        Escala y máximo orden individual disponible. Los filtros 1-D tienen
        longitud ``N + 1``.
    D : int
        Para ``"triangle"``, máximo orden total ``m+n`` (hasta ``2*N``). Para
        ``"square"``, máximo por eje, limitado internamente por ``N``.
    T : int, default=1
        Paso espacial de sampling. ``T=1`` conserva una muestra por posición;
        valores mayores realizan subsampling.
    coefficient_region : {"triangle", "square"}, default="triangle"
        Selección triangular por orden total o cuadrada por orden en cada eje.
    shape : {"full", "symm"}, default="full"
        ``"full"`` conserva toda la convolución. ``"symm"`` extiende ambos
        ejes por reflexión incluyendo la muestra de borde y conserva después
        la región válida.

    Returns
    -------
    coefficients : ndarray
        Stack ``float64`` con los canales en el último eje. Para ``"full"``,
        su shape espacial es ``(ceil((rows+N)/T), ceil((columns+N)/T))``. En
        ``"symm"`` se obtiene de la región válida de la imagen extendida.

    Notes
    -----
    Los filtros 2-D se forman por separabilidad. El primer índice ``m``
    corresponde al eje horizontal y ``n`` al vertical.
    """

    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError("La entrada debe ser una imagen grayscale 2-D.")
    if 0 in values.shape:
        raise ValueError("La imagen no puede tener dimensiones vacías.")
    try:
        values = values.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("La imagen debe contener valores numéricos.") from exc

    T = _sampling_step(T)
    orders = dhtord(N, D, coefficient_region)
    N = int(N)
    if not isinstance(shape, str) or shape.lower() not in {"full", "symm"}:
        raise ValueError('shape debe ser "full" o "symm".')
    shape = shape.lower()
    max_individual_order = max(max(order) for order in orders)
    filters = dhtmtx(N, max_individual_order)

    work = values if shape == "full" else _symmetric_extension(values, N)
    convolution_mode = "full" if shape == "full" else "valid"

    channels = []
    for m, n in orders:
        kernel = np.outer(filters[:, n], filters[:, m])
        channels.append(convolve2d(work, kernel, mode=convolution_mode)[::T, ::T])
    return np.stack(channels, axis=-1).astype(np.float64, copy=False)


__all__ = ["dht2"]
