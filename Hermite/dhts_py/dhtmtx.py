"""Banco unidimensional de filtros discretos de Hermite.

Este módulo construye los filtros binomiales/Krawtchouk usados en el
análisis y, opcionalmente, sus filtros duales para la síntesis con sampling.
La función pública es :func:`dhtmtx`.
"""

import numpy as np


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} debe ser un entero >= 0.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} debe ser un entero >= 0.")
    return value


def dhtmtx(N: int, D: int | None = None, T: int | None = None):
    """Construye el filter bank discreto de Hermite 1-D.

    Parameters
    ----------
    N : int
        Parámetro de escala, entero mayor o igual que cero. Cada filtro tiene
        longitud ``N + 1`` y existen órdenes individuales de ``0`` a ``N``.
    D : int, optional
        Mayor orden 1-D solicitado. Se limita a ``min(D, N)``. Si se omite,
        se construyen los ``N + 1`` filtros disponibles.
    T : int, optional
        Paso de sampling usado para calcular los filtros duales. Debe ser un
        entero mayor o igual que uno. Si se omite, sólo se devuelve el banco
        de análisis.

    Returns
    -------
    H : ndarray, shape (N + 1, min(D, N) + 1)
        Filtros de análisis en ``float64``; la columna ``j`` es el orden ``j``.
    G : ndarray, optional
        Filtros duales de síntesis. Sólo se devuelve cuando ``T`` no es
        ``None`` y tiene la misma forma que ``H``.

    Notes
    -----
    La columna de orden cero es una ventana binomial. Los órdenes superiores
    forman su contraparte discreta, relacionada con polinomios de Krawtchouk
    y con el comportamiento Hermite--Gaussiano continuo.
    """

    N = _nonnegative_integer("N", N)
    max_order = N if D is None else min(_nonnegative_integer("D", D), N)

    if N == 0:
        analysis = np.ones((1, 1), dtype=np.float64)
    else:
        seed = np.array([[0.5, 0.5], [0.5, -0.5]], dtype=np.float64)
        analysis = seed.copy() if max_order > 0 else seed[:, :1].copy()

        for _ in range(2, N + 1):
            number_of_orders = analysis.shape[1]
            smoothed = np.column_stack(
                [
                    np.convolve(seed[:, 0], analysis[:, order], mode="full")
                    for order in range(number_of_orders)
                ]
            )
            if number_of_orders <= max_order:
                highest = np.convolve(seed[:, 1], analysis[:, -1], mode="full")
                analysis = np.column_stack((smoothed, highest))
            else:
                analysis = smoothed

        normalization = 2.0 ** (N / 2.0) * np.sqrt(analysis[: max_order + 1, 0])
        analysis *= normalization[None, :]

    if T is None:
        return analysis

    if isinstance(T, (bool, np.bool_)) or not isinstance(T, (int, np.integer)) or int(T) < 1:
        raise ValueError("T debe ser un entero >= 1.")
    T = int(T)

    # La suma de traslaciones de la ventana de orden cero proporciona los
    # pesos duales que compensan el sampling durante la síntesis.
    sample_positions = np.arange(N % T, 2 * N + 1, T, dtype=int)
    offsets = sample_positions[:, None] - np.arange(N + 1, dtype=int)[None, :]
    valid = (offsets >= 0) & (offsets <= N)
    binomial_window = analysis[:, 0]
    weights = np.sum(np.where(valid, binomial_window[np.clip(offsets, 0, N)], 0.0), axis=0)
    if np.any(weights == 0.0):
        raise ValueError("T no produce pesos de síntesis válidos para este banco.")

    synthesis = analysis[::-1, :] / weights[:, None]
    return analysis, synthesis


__all__ = ["dhtmtx"]
