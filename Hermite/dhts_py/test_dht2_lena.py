"""Prueba equivalente al ejemplo MATLAB de DHT2 sobre Lena.

Equivalencia directa del código MATLAB::

    Y  = dht2(X, N, D, T, 'symm');
    Yr = dht2(X, N, D, T, 'symm', 'r', 'grad');
    Yd = dht2(X, N, D, T, 'symm', 'd', 'hess');

El script abre tres figuras independientes mediante ``dhtshow``, igual que
las tres llamadas finales del ejemplo original. También permite guardarlas
como PNG mediante ``--save-dir``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Permite tanto ``python -m dhts_py.test_dht2_lena`` como
# ``python dhts_py/test_dht2_lena.py`` desde cualquier directorio.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dhts_py.dht2 import dht2
from dhts_py.dhtshow import dhtshow


def cargar_imagen_matlab(ruta: str | Path) -> np.ndarray:
    """Reproduce ``double(imread(ruta))/255`` y el posterior ``rgb2gray``."""

    with Image.open(ruta) as imagen_pil:
        imagen = np.asarray(imagen_pil)

    # El ejemplo MATLAB divide explícitamente entre 255.
    X = imagen.astype(np.float64) / 255.0

    if X.ndim == 3:
        if X.shape[2] >= 3:
            # Coeficientes de luminancia usados por rgb2gray para RGB.
            X = (
                0.298936021293775 * X[..., 0]
                + 0.587043074451121 * X[..., 1]
                + 0.114020904255103 * X[..., 2]
            )
        else:
            # Imagen gris + alfa: conserva el canal de intensidad.
            X = X[..., 0]

    if X.ndim != 2:
        raise ValueError(f"La imagen debe ser 2-D después de rgb2gray; forma obtenida: {X.shape}")
    return X


def _crear_figura(coeficientes: np.ndarray, N: int, D: int, titulo: str):
    """Crea una figura con la misma composición de coeficientes de dhtshow."""

    mosaico = dhtshow(coeficientes, N, D)
    figura, eje = plt.subplots(num=titulo, figsize=(8, 8))
    eje.imshow(mosaico, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    eje.set_title(titulo)
    eje.axis("off")
    figura.tight_layout()
    return figura, mosaico


def ejecutar_prueba(
    ruta_imagen: str | Path | None = None,
    *,
    N: int = 8,
    D: int = 2,
    T: int = 2,
    guardar_en: str | Path | None = None,
    mostrar: bool = True,
):
    """Ejecuta las tres transformadas y genera sus figuras.

    Returns
    -------
    dict
        Contiene la imagen ``X``, los coeficientes ``Y``, ``Yr`` y ``Yd``,
        los tres mosaicos de ``dhtshow`` y las figuras Matplotlib.
    """

    if ruta_imagen is None:
        ruta_imagen = Path(__file__).with_name("lena.jpg")
    ruta_imagen = Path(ruta_imagen).expanduser().resolve()
    if not ruta_imagen.is_file():
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    # 1. Equivalente a: X = double(imread(...))/255; rgb2gray si procede.
    X = cargar_imagen_matlab(ruta_imagen)

    # 2 y 3. Mismos parámetros y mismas llamadas que en MATLAB.
    Y = dht2(X, N, D, T, "symm")
    Yr = dht2(X, N, D, T, "symm", "r", "grad")
    Yd = dht2(X, N, D, T, "symm", "d", "hess")

    # 4. Tres figuras separadas, una por cada llamada a dhtshow.
    fig_cart, mosaico_cart = _crear_figura(Y, N, D, "DHT cartesiana")
    fig_rot, mosaico_rot = _crear_figura(Yr, N, D, "DHT rotada (grad)")
    fig_dir, mosaico_dir = _crear_figura(Yd, N, D, "DHT multidireccional (hess)")

    if guardar_en is not None:
        carpeta = Path(guardar_en).expanduser().resolve()
        carpeta.mkdir(parents=True, exist_ok=True)
        fig_cart.savefig(carpeta / "dht_cartesiana.png", dpi=150, bbox_inches="tight")
        fig_rot.savefig(carpeta / "dht_rotada_grad.png", dpi=150, bbox_inches="tight")
        fig_dir.savefig(carpeta / "dht_multidireccional_hess.png", dpi=150, bbox_inches="tight")

        # Guarda además los mosaicos sin título, márgenes ni reescalado visual.
        plt.imsave(carpeta / "mosaico_dht_cartesiana.png", mosaico_cart, cmap="gray", vmin=0, vmax=1)
        plt.imsave(carpeta / "mosaico_dht_rotada_grad.png", mosaico_rot, cmap="gray", vmin=0, vmax=1)
        plt.imsave(
            carpeta / "mosaico_dht_multidireccional_hess.png",
            mosaico_dir,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

    print(f"Imagen: {ruta_imagen}")
    print(f"X:  {X.shape}")
    print(f"Y:  {Y.shape}  cartesiana")
    print(f"Yr: {Yr.shape}  rotada según gradiente")
    print(f"Yd: {Yd.shape}  multidireccional según Hessiano")

    if mostrar:
        plt.show()

    return {
        "X": X,
        "Y": Y,
        "Yr": Yr,
        "Yd": Yd,
        "mosaicos": (mosaico_cart, mosaico_rot, mosaico_dir),
        "figuras": (fig_cart, fig_rot, fig_dir),
    }


def _argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualiza las DHT cartesianas, rotadas y multidireccionales de Lena."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).with_name("lena.jpg"),
        help="Imagen de entrada (por defecto, lena.jpg incluida en dhts_py).",
    )
    parser.add_argument("--save-dir", type=Path, help="Carpeta opcional para guardar las figuras PNG.")
    parser.add_argument("--no-show", action="store_true", help="No abre las ventanas de Matplotlib.")
    return parser.parse_args()


if __name__ == "__main__":
    argumentos = _argumentos()
    ejecutar_prueba(
        argumentos.image,
        guardar_en=argumentos.save_dir,
        mostrar=not argumentos.no_show,
    )

