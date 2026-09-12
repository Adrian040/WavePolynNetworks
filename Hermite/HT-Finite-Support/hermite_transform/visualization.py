"""Guardado de figuras y métricas sin mezclarlo con el núcleo matemático."""

import csv
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray
Order = tuple[int, int]


def _output_path(path: str | Path) -> Path:
    """Crea el directorio padre y devuelve la ruta como ``Path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_coefficients_grid(
    coefficients: Mapping[Order, Array],
    title: str,
    output_path: str | Path,
    cmap: str = "gray",
) -> None:
    """Guarda los mapas ``L_mn`` en una cuadrícula por orden.

    Parameters
    ----------
    coefficients : mapping
        Mapas 2-D indexados por ``(m, n)``.
    title : str
        Título general de la figura.
    output_path : str or Path
        Archivo de destino.
    cmap : str, default="gray"
        Colormap de Matplotlib.

    Returns
    -------
    None
        Guarda y cierra la figura.
    """
    orders = list(coefficients)
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
            if (m, n) in coefficients:
                value = np.asarray(coefficients[(m, n)])
                limit = max(float(np.max(np.abs(value))), np.finfo(float).eps)
                axis.imshow(value, cmap=cmap, vmin=-limit, vmax=limit)
                axis.set_title(rf"$L_{{{m},{n}}}$")
            axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(_output_path(output_path), dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_theta_image(theta: Array, output_path: str | Path) -> None:
    """Guarda un mapa de orientación en grados.

    Parameters
    ----------
    theta : ndarray
        Mapa angular en radianes.
    output_path : str or Path
        Archivo de destino.

    Returns
    -------
    None
        Usa un colormap cíclico y cierra la figura.
    """
    figure, axis = plt.subplots(figsize=(7, 6))
    shown = axis.imshow(
        np.rad2deg(theta),
        cmap="twilight",
        vmin=-180,
        vmax=180,
    )
    axis.set_title(r"Dirección local del gradiente $\theta$ [grados]")
    axis.axis("off")
    figure.colorbar(shown, ax=axis)
    figure.tight_layout()
    figure.savefig(_output_path(output_path), dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_intensity_image(
    image: Array,
    title: str,
    output_path: str | Path,
) -> None:
    """Guarda una matriz 2-D como imagen grayscale.

    Parameters
    ----------
    image : ndarray
        Imagen o resumen escalar.
    title : str
        Título de la figura.
    output_path : str or Path
        Archivo de destino.

    Returns
    -------
    None
        La normalización visual no altera el array original.
    """
    image = np.asarray(image)
    lower, upper = float(np.min(image)), float(np.max(image))
    if lower == upper:
        upper = lower + 1.0
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(image, cmap="gray", vmin=lower, vmax=upper)
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(_output_path(output_path), dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_reconstruction_comparison(
    original: Array,
    reconstructed: Array,
    output_path: str | Path,
) -> None:
    """Guarda original, reconstrucción y error absoluto.

    Parameters
    ----------
    original, reconstructed : ndarray
        Imágenes con la misma forma.
    output_path : str or Path
        Archivo de destino.

    Returns
    -------
    None
        Guarda una figura de tres paneles.
    """
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)
    error = np.abs(original - reconstructed)
    lower = float(min(np.min(original), np.min(reconstructed)))
    upper = float(max(np.max(original), np.max(reconstructed)))
    if lower == upper:
        upper = lower + 1.0

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(original, cmap="gray", vmin=lower, vmax=upper)
    axes[0].set_title("Original")
    axes[1].imshow(reconstructed, cmap="gray", vmin=lower, vmax=upper)
    axes[1].set_title("Reconstrucción truncada")
    shown = axes[2].imshow(error, cmap="gray", vmin=0.0)
    axes[2].set_title(r"Error absoluto $|I-\hat I|$")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(shown, ax=axes[2], fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(_output_path(output_path), dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_metrics_csv(metrics: Mapping[str, float], output_path: str | Path) -> None:
    """Guarda métricas en un CSV de columnas ``metric`` y ``value``.

    Parameters
    ----------
    metrics : mapping of str to float
        Valores escalares que se escribirán.
    output_path : str or Path
        Archivo CSV de destino.

    Returns
    -------
    None
        Escribe el archivo con codificación UTF-8.
    """
    with _output_path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())
