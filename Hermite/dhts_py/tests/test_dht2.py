"""Genera resultados visuales de DHT cartesiana, RDHT y DDHT.

El script usa ``N=8``, ``D=3``, ``T=2``, bordes ``symm`` y región
triangular. Todos los resultados se escriben en ``tests/resultados/dht``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dhts_py import ddht, dht2, dhtord, gauge, rdht
from dhts_py.utils import coefficient_index, plot_coefficients


TESTS_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = TESTS_DIR / "data" / "lena.jpg"
RESULTS_DIR = TESTS_DIR / "resultados" / "dht"


def load_grayscale_image(path: str | Path) -> np.ndarray:
    """Carga una imagen, la escala por 255 y convierte RGB a grayscale."""

    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(
            f"No se encontró la imagen {image_path}. Coloque lena.jpg en "
            f"{DEFAULT_IMAGE} o use --image RUTA."
        )

    with Image.open(image_path) as image:
        pixels = np.asarray(image)
    values = pixels.astype(np.float64) / 255.0
    if values.ndim == 3 and values.shape[2] >= 3:
        values = (
            0.298936021293775 * values[..., 0]
            + 0.587043074451121 * values[..., 1]
            + 0.114020904255103 * values[..., 2]
        )
    elif values.ndim == 3:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"La imagen debe producir un array grayscale 2-D; se obtuvo {values.shape}.")
    return values


def coefficient_montage(
    coefficients: np.ndarray,
    orders: list[tuple[int, int]],
) -> np.ndarray:
    """Compone filas ``n`` y columnas ``m`` con la escala visual indicada."""

    values = np.asarray(coefficients, dtype=np.float64)
    height, width = values.shape[:2]
    max_m = max(m for m, _ in orders)
    max_n = max(n for _, n in orders)
    montage = np.ones(((max_n + 1) * height, (max_m + 1) * width), dtype=np.float64)

    for channel, (m, n) in enumerate(orders):
        coefficient_map = values[..., channel]
        if (m, n) == (0, 0):
            displayed = np.clip((coefficient_map - 0.1) / 0.8, 0.0, 1.0)
        else:
            displayed = np.clip((coefficient_map + 0.2) / 0.4, 0.0, 1.0)
        montage[n * height : (n + 1) * height, m * width : (m + 1) * width] = displayed
    return montage


def _save_labeled(
    coefficients: np.ndarray,
    orders: list[tuple[int, int]],
    path: Path,
    show: bool,
) -> None:
    figure, _ = plot_coefficients(
        coefficients,
        orders,
        coefficient_region="triangle",
        normalize="individual",
    )
    figure.savefig(path, dpi=120, bbox_inches="tight")
    if not show:
        plt.close(figure)


def _save_angle(theta: np.ndarray, title: str, path: Path, show: bool) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(theta, cmap="hsv", vmin=-np.pi, vmax=np.pi)
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(image, ax=axis, label="radianes")
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    if not show:
        plt.close(figure)


def main(image_path: str | Path = DEFAULT_IMAGE, *, show: bool = False) -> dict[str, np.ndarray]:
    """Ejecuta las tres transformaciones y guarda sus comparaciones visuales."""

    N, D, T = 8, 3, 2
    shape = "symm"
    coefficient_region = "triangle"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X = load_grayscale_image(image_path)
    orders = dhtord(N=N, D=D, coefficient_region=coefficient_region)
    Y = dht2(
        X,
        N=N,
        D=D,
        T=T,
        shape=shape,
        coefficient_region=coefficient_region,
    )
    if Y.shape[-1] != 10:
        raise AssertionError(f"Se esperaban 10 coefficient maps; se obtuvieron {Y.shape[-1]}.")

    theta_grad = gauge(Y, N=N, D=D, mode="gradient", coefficient_region=coefficient_region)
    Yr = rdht(
        Y,
        theta_grad,
        N=N,
        D=D,
        direction="forward",
        coefficient_region=coefficient_region,
    )
    Yr_reference_display = Yr.copy()
    Yr_reference_display[..., coefficient_index(orders, (0, 1))] = theta_grad / (2.0 * np.pi)

    theta_hess = gauge(Y, N=N, D=D, mode="hessian", coefficient_region=coefficient_region)
    Yd = ddht(
        Y,
        theta_hess,
        N=N,
        D=D,
        direction="forward",
        coefficient_region=coefficient_region,
    )
    Yd_reference_display = Yd.copy()
    Yd_reference_display[..., coefficient_index(orders, (1, 1))] = theta_hess / (2.0 * np.pi)

    plt.imsave(RESULTS_DIR / "original.png", X, cmap="gray", vmin=0.0, vmax=1.0)

    cartesian_montage = coefficient_montage(Y, orders)
    rotated_clean_montage = coefficient_montage(Yr, orders)
    rotated_reference_montage = coefficient_montage(Yr_reference_display, orders)
    directional_clean_montage = coefficient_montage(Yd, orders)
    directional_reference_montage = coefficient_montage(Yd_reference_display, orders)

    montage_files = {
        "cartesian_montage.png": cartesian_montage,
        "rotated_montage.png": rotated_reference_montage,
        "rotated_montage_without_theta_replacement.png": rotated_clean_montage,
        "rotated_montage_with_theta_L01.png": rotated_reference_montage,
        "rotated_coefficients_clean.png": rotated_clean_montage,
        "rotated_coefficients_reference_display.png": rotated_reference_montage,
        "directional_montage.png": directional_reference_montage,
        "directional_montage_without_theta_replacement.png": directional_clean_montage,
        "directional_montage_with_theta_L11.png": directional_reference_montage,
        "directional_coefficients_clean.png": directional_clean_montage,
        "directional_coefficients_reference_display.png": directional_reference_montage,
    }
    for filename, montage in montage_files.items():
        plt.imsave(RESULTS_DIR / filename, montage, cmap="gray", vmin=0.0, vmax=1.0)

    _save_labeled(Y, orders, RESULTS_DIR / "cartesian_labeled.png", show)
    _save_labeled(Yr_reference_display, orders, RESULTS_DIR / "rotated_labeled.png", show)
    _save_labeled(
        Yr,
        orders,
        RESULTS_DIR / "rotated_labeled_without_theta_replacement.png",
        show,
    )
    _save_labeled(
        Yr_reference_display,
        orders,
        RESULTS_DIR / "rotated_labeled_with_theta_L01.png",
        show,
    )
    _save_labeled(Yd_reference_display, orders, RESULTS_DIR / "directional_labeled.png", show)
    _save_labeled(
        Yd,
        orders,
        RESULTS_DIR / "directional_labeled_without_theta_replacement.png",
        show,
    )
    _save_labeled(
        Yd_reference_display,
        orders,
        RESULTS_DIR / "directional_labeled_with_theta_L11.png",
        show,
    )
    _save_angle(theta_grad, "Orientación por gradient", RESULTS_DIR / "theta_gradient.png", show)
    _save_angle(theta_hess, "Orientación por Hessian", RESULTS_DIR / "theta_hessian.png", show)

    arrays = {
        "Y.npy": Y,
        "Yr.npy": Yr,
        "Yr_reference_display.npy": Yr_reference_display,
        "Yd.npy": Yd,
        "Yd_reference_display.npy": Yd_reference_display,
        "theta_gradient.npy": theta_grad,
        "theta_hessian.npy": theta_hess,
    }
    for filename, array in arrays.items():
        np.save(RESULTS_DIR / filename, array)

    print(f"Imagen: {Path(image_path).expanduser().resolve()}")
    print(f"Parámetros: N={N}, D={D}, T={T}, shape={shape}, region={coefficient_region}")
    print(f"X:  {X.shape}")
    print(f"Y:  {Y.shape} cartesiana")
    print(f"Yr: {Yr.shape} RDHT limpia; theta separado")
    print(f"Yd: {Yd.shape} DDHT limpia; theta separado")
    print(f"Resultados: {RESULTS_DIR}")

    if show:
        plt.show()
    return {
        "X": X,
        "Y": Y,
        "Yr": Yr,
        "Yd": Yd,
        "theta_gradient": theta_grad,
        "theta_hessian": theta_hess,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera comparaciones visuales DHT/RDHT/DDHT.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Imagen de entrada.")
    parser.add_argument("--show", action="store_true", help="Muestra las figuras al finalizar.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    main(arguments.image, show=arguments.show)
