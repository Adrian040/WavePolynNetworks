"""Lectura consistente de imágenes para la transformada de Hermite."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


Array = np.ndarray
_RGB_WEIGHTS = np.asarray(
    [0.298936021293775, 0.587043074451121, 0.114020904255103],
    dtype=np.float64,
)


def _array_to_grayscale(image: Array, source: str) -> Array:
    """Convierte una imagen numérica a grayscale ``float64`` sin reescalarla."""
    array = np.asarray(image)
    if np.iscomplexobj(array) or not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"{source} debe contener datos numéricos reales.")

    if array.ndim == 2:
        grayscale = array
    elif array.ndim == 3 and array.shape[-1] in {1, 2}:
        grayscale = array[..., 0]
    elif array.ndim == 3 and array.shape[-1] in {3, 4}:
        grayscale = np.tensordot(
            array[..., :3].astype(np.float64, copy=False),
            _RGB_WEIGHTS,
            axes=([-1], [0]),
        )
    else:
        raise ValueError(
            f"{source} debe tener forma (alto, ancho) o entre 1 y 4 canales."
        )

    result = np.asarray(grayscale, dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{source} debe ser una imagen no vacía con valores finitos.")
    return result


def _pil_to_grayscale(image: Image.Image, source: str) -> Array:
    """Aplica orientación EXIF e interpreta correctamente el modo de PIL."""
    oriented = ImageOps.exif_transpose(image)
    mode = oriented.mode
    direct_modes = {"1", "L", "LA", "I", "F", "RGB", "RGBA", "RGBX"}
    array = (
        np.asarray(oriented)
        if mode in direct_modes or mode.startswith("I;16")
        else np.asarray(oriented.convert("RGB"))
    )
    return _array_to_grayscale(array, f"{source} (modo PIL {mode!r})")


def read_image(image: str | Path | Image.Image | Array) -> Array:
    """Lee una imagen como una matriz 2-D ``float64``.

    Parameters
    ----------
    image : str, Path, PIL.Image.Image or ndarray
        Imagen grayscale, LA, RGB o RGBA. Las rutas respetan EXIF y conservan
        la profundidad de bits, incluidos TIFF grayscale de 16 bits.

    Returns
    -------
    ndarray
        Imagen bidimensional en su escala original; no se normaliza a 0--1.
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as pil_image:
            return _pil_to_grayscale(pil_image, f"archivo {str(image)!r}")
    if isinstance(image, Image.Image):
        return _pil_to_grayscale(image, "imagen PIL")
    return _array_to_grayscale(image, "array de entrada")
