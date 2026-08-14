"""Prueba equivalente al script MATLAB de DHT2 directa e inversa.

La secuencia reproducida es::

    Y  = dht2(X, N, D, T, 'symm')
    Yr = dht2(X, N, D, T, 'symm', 'r', 'grad')
    Yd = dht2(X, N, D, T, 'symm', 'd', 'hess')

    X_rec   = idht2(Y,  xsiz, N, D, T, 'symm')
    X_rec_r = idht2(Yr, xsiz, N, D, T, 'symm', 'r')
    X_rec_d = idht2(Yd, xsiz, N, D, T, 'symm', 'd')

Por fidelidad, la llamada inversa direccional no recibe ``'hess'``: así está
escrita en el script MATLAB proporcionado y así se ejecuta aquí.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Admite tanto ``python -m dhts_py.test_idht_lena`` como la ejecución directa
# ``python dhts_py/test_idht_lena.py`` desde cualquier directorio.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dhts_py.dht2 import dht2
from dhts_py.idht2 import idht2
from dhts_py.test_dht2_lena import cargar_imagen_matlab


def _mostrar_como_matlab(eje, imagen: np.ndarray, titulo: str) -> None:
    """Equivalente a ``imshow(imagen, []); title(titulo)``."""

    eje.imshow(imagen, cmap="gray", interpolation="nearest")
    eje.set_title(titulo)
    eje.axis("off")


def ejecutar_prueba_idht(
    ruta_imagen: str | Path | None = None,
    *,
    N: int = 8,
    D: int = 2,
    T: int = 2,
    guardar_en: str | Path | None = None,
    mostrar: bool = True,
):
    """Ejecuta las transformadas y reconstrucciones del script MATLAB.

    Parameters
    ----------
    ruta_imagen:
        Imagen de entrada. Por defecto utiliza ``lena.jpg`` incluida en el
        paquete.
    N, D, T:
        Parámetros DHT. Sus valores predeterminados son exactamente los del
        script MATLAB: ``N=8``, ``D=2`` y ``T=2``.
    guardar_en:
        Carpeta opcional en la que se guardan la figura y los arreglos NumPy.
    mostrar:
        Si es verdadero, abre la ventana de Matplotlib.

    Returns
    -------
    dict
        Imagen original, coeficientes, reconstrucciones y figura generada.
    """

    plt.close("all")  # Equivalente a close all.

    if ruta_imagen is None:
        ruta_imagen = Path(__file__).with_name("lena.jpg")
    ruta_imagen = Path(ruta_imagen).expanduser().resolve()
    if not ruta_imagen.is_file():
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    # 1. X = double(imread(img_name))/255; rgb2gray si corresponde.
    print(f"img_name = {ruta_imagen.name}")
    X = cargar_imagen_matlab(ruta_imagen)
    xsiz = X.shape

    # 2 y 3. Transformadas directas con los mismos parámetros y opciones.
    print("Calculando transformadas directas...")
    Y = dht2(X, N, D, T, "symm")
    Yr = dht2(X, N, D, T, "symm", "r", "grad")
    Yd = dht2(X, N, D, T, "symm", "d", "hess")

    # 4. Transformadas inversas: llamadas idénticas a las del script MATLAB.
    print("Reconstruyendo imágenes...")
    X_rec = idht2(Y, xsiz, N, D, T, "symm")
    X_rec_r = idht2(Yr, xsiz, N, D, T, "symm", "r")
    X_rec_d = idht2(Yd, xsiz, N, D, T, "symm", "d")

    # 5. Equivalente a figure(...); subplot(1,4,k); imshow(...,[]).
    print("Generando visualización...")
    figura, ejes = plt.subplots(1, 4, figsize=(10, 4), num="Comparativa de Reconstrucción")
    try:
        figura.canvas.manager.set_window_title("Comparativa de Reconstrucción")
    except (AttributeError, NotImplementedError):
        pass

    _mostrar_como_matlab(ejes[0], X, "Original")
    _mostrar_como_matlab(ejes[1], X_rec, "Rec. Estándar (Y)")
    _mostrar_como_matlab(ejes[2], X_rec_r, "Rec. Rotada (Yr)")
    _mostrar_como_matlab(ejes[3], X_rec_d, "Rec. Direccional (Yd)")
    figura.tight_layout()

    if guardar_en is not None:
        carpeta = Path(guardar_en).expanduser().resolve()
        carpeta.mkdir(parents=True, exist_ok=True)
        figura.savefig(carpeta / "comparativa_reconstruccion.png", dpi=150, bbox_inches="tight")
        np.savez_compressed(
            carpeta / "resultados_idht_lena.npz",
            X=X,
            Y=Y,
            Yr=Yr,
            Yd=Yd,
            X_rec=X_rec,
            X_rec_r=X_rec_r,
            X_rec_d=X_rec_d,
            N=N,
            D=D,
            T=T,
        )

    if mostrar:
        plt.show()

    return {
        "X": X,
        "xsiz": xsiz,
        "Y": Y,
        "Yr": Yr,
        "Yd": Yd,
        "X_rec": X_rec,
        "X_rec_r": X_rec_r,
        "X_rec_d": X_rec_d,
        "figura": figura,
    }


def _argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prueba DHT2 directa/inversa equivalente al script MATLAB suministrado."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).with_name("lena.jpg"),
        help="Imagen de entrada (por defecto, dhts_py/lena.jpg).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Carpeta opcional para guardar la figura y resultados_idht_lena.npz.",
    )
    parser.add_argument("--no-show", action="store_true", help="No abre la ventana de Matplotlib.")
    return parser.parse_args()


if __name__ == "__main__":
    argumentos = _argumentos()
    ejecutar_prueba_idht(
        argumentos.image,
        guardar_en=argumentos.save_dir,
        mostrar=not argumentos.no_show,
    )

