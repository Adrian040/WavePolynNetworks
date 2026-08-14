"""Genera comparaciones visuales de reconstrucción DHT truncada y completa.

Todos los resultados se guardan en ``tests/resultados/idht``. El experimento
triangular es truncado por diseño; el square completo debe reconstruir hasta
precisión de punto flotante.
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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dhts_py import dht2, gauge, idht2, rdht
from dhts_py.tests.test_dht2 import DEFAULT_IMAGE, load_grayscale_image
from dhts_py.utils import plot_reconstruction_comparison, reconstruction_metrics


TESTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TESTS_DIR / "resultados" / "idht"


def _save_experiment(
    prefix: str,
    original: np.ndarray,
    reconstructed: np.ndarray,
    show: bool,
) -> dict[str, float]:
    error = np.abs(original - reconstructed)
    plt.imsave(RESULTS_DIR / f"{prefix}_original.png", original, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imsave(
        RESULTS_DIR / f"{prefix}_reconstruction.png",
        reconstructed,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    plt.imsave(
        RESULTS_DIR / f"{prefix}_absolute_error.png",
        error,
        cmap="magma",
        vmin=0.0,
        vmax=max(float(np.max(error)), np.finfo(np.float64).eps),
    )
    figure, _ = plot_reconstruction_comparison(original, reconstructed, data_range=1.0)
    figure.savefig(RESULTS_DIR / f"{prefix}_comparison.png", dpi=140, bbox_inches="tight")
    if not show:
        plt.close(figure)
    return reconstruction_metrics(original, reconstructed, data_range=1.0)


def _print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(name)
    for key in ("mse", "rmse", "mae", "max_abs_error", "psnr"):
        print(f"  {key}: {metrics[key]:.16g}")


def main(image_path: str | Path = DEFAULT_IMAGE, *, show: bool = False) -> dict[str, object]:
    """Ejecuta las reconstrucciones truncada, completa y posterior a RDHT."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X = load_grayscale_image(image_path)

    # Experimento A: sólo m+n <= 3; no se espera una reconstrucción perfecta.
    Y = dht2(X, N=8, D=3, T=2, shape="symm", coefficient_region="triangle")
    X_truncated = idht2(
        Y,
        X.shape,
        N=8,
        D=3,
        T=2,
        shape="symm",
        coefficient_region="triangle",
    )
    truncated_metrics = _save_experiment("truncated", X, X_truncated, show)

    # Experimento B: los 81 coefficient maps del banco bidimensional completo.
    Y_full = dht2(X, N=8, D=8, T=2, shape="full", coefficient_region="square")
    if Y_full.shape[-1] != 81:
        raise AssertionError(f"Se esperaban 81 coefficient maps; se obtuvieron {Y_full.shape[-1]}.")
    X_full = idht2(
        Y_full,
        X.shape,
        N=8,
        D=8,
        T=2,
        shape="full",
        coefficient_region="square",
    )
    full_metrics = _save_experiment("full", X, X_full, show)
    if not np.allclose(X_full, X, atol=5e-13, rtol=5e-13):
        raise AssertionError("La reconstrucción full square no alcanzó precisión numérica.")

    # Consistencia: DHT -> RDHT -> RDHT inversa -> IDHT truncada.
    theta = gauge(Y, N=8, D=3, mode="gradient", coefficient_region="triangle")
    rotated = rdht(Y, theta, N=8, D=3, direction="forward", coefficient_region="triangle")
    recovered = rdht(
        rotated,
        theta,
        N=8,
        D=3,
        direction="inverse",
        coefficient_region="triangle",
    )
    X_from_recovered = idht2(
        recovered,
        X.shape,
        N=8,
        D=3,
        T=2,
        shape="symm",
        coefficient_region="triangle",
    )
    coefficient_roundtrip_error = float(np.max(np.abs(Y - recovered)))
    reconstruction_roundtrip_error = float(np.max(np.abs(X_truncated - X_from_recovered)))
    plt.imsave(
        RESULTS_DIR / "rdht_recovered_reconstruction.png",
        X_from_recovered,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    print(f"Imagen: {Path(image_path).expanduser().resolve()}")
    print(f"Y triangle: {Y.shape}")
    print(f"Y full square: {Y_full.shape}")
    _print_metrics("Triangle truncado — N=8, D=3, T=2, symm", truncated_metrics)
    _print_metrics("Square completo — N=8, D=8, T=2, full", full_metrics)
    print(f"RDHT max|Y - recovered|: {coefficient_roundtrip_error:.16g}")
    print(f"RDHT max|X_truncated - X_from_recovered|: {reconstruction_roundtrip_error:.16g}")
    print(f"Resultados: {RESULTS_DIR}")

    if show:
        plt.show()
    return {
        "X": X,
        "Y": Y,
        "X_truncated": X_truncated,
        "truncated_metrics": truncated_metrics,
        "Y_full": Y_full,
        "X_full": X_full,
        "full_metrics": full_metrics,
        "coefficient_roundtrip_error": coefficient_roundtrip_error,
        "reconstruction_roundtrip_error": reconstruction_roundtrip_error,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera comparaciones visuales de IDHT.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Imagen de entrada.")
    parser.add_argument("--show", action="store_true", help="Muestra las figuras al finalizar.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    main(arguments.image, show=arguments.show)
