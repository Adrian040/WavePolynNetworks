"""
hermite_transform_rotada.py

Implementación práctica de la Transformada de Hermite normal y rotada/dirigida.

La función principal es:

    hermite_transform_image(...)

Sirve para:
1) Calcular coeficientes de Hermite normales.
2) Calcular coeficientes de Hermite rotados a un ángulo fijo.
3) Calcular coeficientes de Hermite rotados hacia la orientación local dominante
   usando la fórmula del artículo del Dr. Boris Escalante-Ramírez:

       tan(theta) = L_{0,1} / L_{1,0}

   En código se usa arctan2(L01, L10), que es la forma numéricamente correcta.

La función devuelve un diccionario con:
- transformed_image: imagen 2D normalizada en [0,1], útil para visualizar o guardar.
- coeff_stack: arreglo HxWxC o CxHxW con los mapas de coeficientes, útil para modelos.
- coeffs: diccionario con cada mapa de coeficientes.
- theta: mapa de orientación usado, si aplica.
- energy: mapa de energía no normalizado.
- saved_path: ruta de guardado, si save_image=True.

Dependencias:
    pip install numpy scipy pillow

Ejemplo rápido:

    from hermite_transform_rotada import hermite_transform_image

    result = hermite_transform_image(
        image="mi_imagen.png",
        max_order=3,
        sigma=2.0,
        use_rotated=True,
        use_dominant_orientation=True,
        save_image=True,
        output_path="hermite_rotada.png"
    )

    transformed = result["transformed_image"]  # HxW, float32, [0,1]
    stack = result["coeff_stack"]              # HxWxC, listo para un modelo
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Dict, Iterable, Literal, Tuple, Union, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

Array = np.ndarray
CoeffKey = Tuple[int, int]
CoeffDict = "OrderedDict[CoeffKey, Array]"
ImageInput = Union[str, Path, Image.Image, Array]


@dataclass
class HermiteResult:
    """Contenedor opcional por si se prefiere usar atributos en vez de dict."""

    transformed_image: Array
    coeff_stack: Array
    coeffs: CoeffDict
    theta: Optional[Array]
    energy: Array
    saved_path: Optional[str]
    params: dict

    def to_dict(self) -> dict:
        return {
            "transformed_image": self.transformed_image,
            "coeff_stack": self.coeff_stack,
            "coeffs": self.coeffs,
            "theta": self.theta,
            "energy": self.energy,
            "saved_path": self.saved_path,
            "params": self.params,
        }


def load_image_gray(image: ImageInput) -> Array:
    """
    Carga una imagen común: png, jpg, jpeg, tif/tiff, bmp, etc.

    Parameters
    ----------
    image:
        Ruta, PIL.Image o np.ndarray.

    Returns
    -------
    img:
        Imagen en escala de grises, np.float64, rango [0,1].
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image)
        img = img.convert("L")
        arr = np.asarray(img, dtype=np.float64)
    elif isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"), dtype=np.float64)
    else:
        arr = np.asarray(image)
        if arr.ndim == 3:
            # Conversión RGB/RGBA a gris con ponderación estándar aproximada.
            arr = arr[..., :3].astype(np.float64)
            arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        else:
            arr = arr.astype(np.float64)

    # Normalización robusta a [0,1]
    if arr.size == 0:
        raise ValueError("La imagen está vacía.")

    if arr.max() > 1.5:
        arr = arr / 255.0

    arr = np.clip(arr, 0.0, 1.0)
    return arr


def normalize01(x: Array, eps: float = 1e-12, robust: bool = True) -> Array:
    """Normaliza un arreglo a [0,1]."""
    x = np.asarray(x, dtype=np.float64)
    if robust:
        lo, hi = np.percentile(x, [1, 99])
    else:
        lo, hi = float(np.min(x)), float(np.max(x))
    y = (x - lo) / (hi - lo + eps)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def save_gray_image(image: Array, output_path: Union[str, Path]) -> str:
    """Guarda una imagen 2D float [0,1] como archivo de imagen."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img8 = (normalize01(image, robust=False) * 255.0).round().astype(np.uint8)
    Image.fromarray(img8, mode="L").save(output_path)
    return str(output_path)


def hermite_orders(max_order: int) -> list[CoeffKey]:
    """
    Devuelve los pares (m,n) con m+n <= max_order.

    Convención:
    - m: orden de derivada en x / columnas.
    - n: orden de derivada en y / filas.
    """
    if max_order < 0:
        raise ValueError("max_order debe ser >= 0.")
    orders = []
    for total in range(max_order + 1):
        for m in range(total + 1):
            n = total - m
            orders.append((m, n))
    return orders


def normal_hermite_coefficients(
    image: ImageInput,
    max_order: int = 3,
    sigma: float = 2.0,
    mode: str = "reflect",
    scale_normalized: bool = True,
) -> CoeffDict:
    """
    Calcula coeficientes de Hermite normales usando derivadas Gaussianas.

    Importante sobre la convención de índices:
    scipy.ndimage.gaussian_filter usa order=(orden_filas_y, orden_columnas_x).
    Aquí guardamos los coeficientes como L_(m,n), donde:
        m = orden en x
        n = orden en y
    Por eso se llama gaussian_filter(..., order=(n,m)).
    """
    if sigma <= 0:
        raise ValueError("sigma debe ser > 0.")

    img = load_image_gray(image)
    coeffs: CoeffDict = OrderedDict()

    for m, n in hermite_orders(max_order):
        c = gaussian_filter(img, sigma=sigma, order=(n, m), mode=mode)

        # Derivadas normalizadas por escala. Ayuda a comparar órdenes distintos.
        if scale_normalized:
            c = c * (sigma ** (m + n))

        coeffs[(m, n)] = c.astype(np.float32)

    return coeffs


def orientation_from_first_order(coeffs: CoeffDict, eps: float = 1e-12) -> Array:
    """
    Calcula la orientación local según:

        tan(theta) = L_{0,1} / L_{1,0}

    Se usa arctan2(L01, L10) en lugar de arctan(L01/L10), porque arctan2
    maneja cuadrantes y evita divisiones problemáticas entre cero.

    Returns
    -------
    theta:
        Mapa HxW de orientación en radianes, restringido a [0, pi).
        Se usa [0, pi) porque para texturas/orientaciones locales normalmente
        theta y theta + pi representan el mismo eje.
    """
    if (1, 0) not in coeffs or (0, 1) not in coeffs:
        raise ValueError("Para calcular theta se requiere max_order >= 1.")

    L10 = coeffs[(1, 0)].astype(np.float64)
    L01 = coeffs[(0, 1)].astype(np.float64)
    theta = np.arctan2(L01, L10 + eps)
    theta = np.mod(theta, np.pi)
    return theta.astype(np.float32)


def _as_theta_map(theta: Union[float, Array], shape: tuple[int, int]) -> Array:
    """Convierte un ángulo escalar o mapa de ángulos a un mapa HxW en radianes."""
    if np.isscalar(theta):
        return np.full(shape, float(theta), dtype=np.float64)
    theta_arr = np.asarray(theta, dtype=np.float64)
    if theta_arr.shape != shape:
        raise ValueError(f"theta debe ser escalar o tener shape {shape}, pero tiene {theta_arr.shape}.")
    return theta_arr


def steer_coefficient(
    coeffs: CoeffDict,
    out_order_x: int,
    out_order_y: int,
    theta: Union[float, Array],
) -> Array:
    """
    Calcula un coeficiente Hermite rotado L_{out_order_x,out_order_y}^{theta}.

    Usa la expansión:

        D_x' =  cos(theta) D_x + sin(theta) D_y
        D_y' = -sin(theta) D_x + cos(theta) D_y

    y expande (D_x')^a (D_y')^b como combinación de derivadas cartesianas.

    Para el caso compacto más usado en el artículo:

        L_{N,0}^{theta} = sum_k C(N,k) cos^(N-k)(theta) sin^k(theta) L_{N-k,k}
    """
    if out_order_x < 0 or out_order_y < 0:
        raise ValueError("Los órdenes deben ser >= 0.")

    shape = next(iter(coeffs.values())).shape
    theta_map = _as_theta_map(theta, shape)
    c = np.cos(theta_map)
    s = np.sin(theta_map)

    a = out_order_x
    b = out_order_y
    out = np.zeros(shape, dtype=np.float64)

    # (D_x')^a = sum_i C(a,i) c^(a-i) s^i D_x^(a-i) D_y^i
    # (D_y')^b = sum_j C(b,j) (-s)^(b-j) c^j D_x^(b-j) D_y^j
    for i in range(a + 1):
        coef_i = comb(a, i) * (c ** (a - i)) * (s ** i)
        dx_i = a - i
        dy_i = i

        for j in range(b + 1):
            coef_j = comb(b, j) * ((-s) ** (b - j)) * (c ** j)
            dx_j = b - j
            dy_j = j

            base_key = (dx_i + dx_j, dy_i + dy_j)
            if base_key not in coeffs:
                raise ValueError(
                    f"Falta el coeficiente base {base_key}. "
                    f"Aumenta max_order al menos a {sum(base_key)}."
                )

            out += coef_i * coef_j * coeffs[base_key]

    return out.astype(np.float32)


def rotated_hermite_coefficients(
    coeffs: CoeffDict,
    theta: Union[float, Array],
    max_order: int,
    compact: bool = True,
    include_dc: bool = True,
) -> CoeffDict:
    """
    Rota los coeficientes de Hermite.

    Parameters
    ----------
    coeffs:
        Coeficientes normales L_(m,n).
    theta:
        Ángulo escalar o mapa HxW de ángulos en radianes.
    max_order:
        Orden máximo.
    compact:
        Si True, devuelve la representación compacta orientada:
            L00, L10^theta, L20^theta, ..., LN0^theta.
        Si False, devuelve todos los coeficientes rotados Lmn^theta con m+n<=N.
    include_dc:
        Si True, conserva L00.
    """
    out: CoeffDict = OrderedDict()

    if include_dc and (0, 0) in coeffs:
        out[(0, 0)] = coeffs[(0, 0)].astype(np.float32)

    if compact:
        for total in range(1, max_order + 1):
            out[(total, 0)] = steer_coefficient(coeffs, total, 0, theta)
    else:
        for m, n in hermite_orders(max_order):
            if (m, n) == (0, 0) and include_dc:
                continue
            out[(m, n)] = steer_coefficient(coeffs, m, n, theta)

    return out


def coefficients_to_stack(
    coeffs: CoeffDict,
    channel_axis: Literal["last", "first"] = "last",
) -> Array:
    """
    Convierte el diccionario de coeficientes en un arreglo para modelos.

    channel_axis='last'  -> HxWxC, común para scikit-image/TensorFlow.
    channel_axis='first' -> CxHxW, común para PyTorch.
    """
    maps = [np.asarray(v, dtype=np.float32) for v in coeffs.values()]
    stack = np.stack(maps, axis=-1)  # HxWxC
    if channel_axis == "first":
        stack = np.moveaxis(stack, -1, 0)  # CxHxW
    elif channel_axis != "last":
        raise ValueError("channel_axis debe ser 'last' o 'first'.")
    return stack.astype(np.float32)


def hermite_energy(coeffs: CoeffDict, include_dc: bool = False) -> Array:
    """
    Calcula la energía local de los coeficientes Hermite.

    Esta energía es útil como imagen transformada 2D para visualización o como
    descriptor local para clasificación de texturas.
    """
    first = next(iter(coeffs.values()))
    energy = np.zeros_like(first, dtype=np.float64)

    for key, value in coeffs.items():
        if key == (0, 0) and not include_dc:
            continue
        v = np.asarray(value, dtype=np.float64)
        energy += v * v

    return energy.astype(np.float32)


def default_output_path(image: ImageInput, suffix: str = "_hermite_transform.png") -> str:
    """Crea una ruta de salida por default."""
    if isinstance(image, (str, Path)):
        p = Path(image)
        return str(p.with_name(p.stem + suffix))
    return "hermite_transform.png"


def hermite_transform_image(
    image: ImageInput,
    max_order: int = 3,
    sigma: float = 2.0,
    use_rotated: bool = True,
    use_dominant_orientation: bool = True,
    angle: float = 0.0,
    angle_unit: Literal["degrees", "radians"] = "degrees",
    compact_rotated: bool = True,
    include_dc_in_coeffs: bool = True,
    include_dc_in_energy: bool = False,
    mode: str = "reflect",
    scale_normalized: bool = True,
    channel_axis: Literal["last", "first"] = "last",
    save_image: bool = True,
    output_path: Optional[Union[str, Path]] = None,
    return_dataclass: bool = False,
) -> Union[dict, HermiteResult]:
    """
    Función principal para usar la Transformada de Hermite normal o rotada.

    Parameters
    ----------
    image:
        Imagen de entrada. Acepta ruta .png/.jpg/.jpeg/.tif/.tiff/.bmp, PIL.Image
        o np.ndarray. Se convierte a escala de grises.

    max_order:
        Orden máximo de la transformada. Ej. 2, 3, 4.

    sigma:
        Escala de la Gaussiana. Controla el tamaño de estructura/textura analizada.

    use_rotated:
        Si False, calcula la transformada Hermite normal.
        Si True, calcula la transformada Hermite rotada.

    use_dominant_orientation:
        Solo se usa si use_rotated=True.
        Si True, calcula theta local con:
            theta = arctan2(L_{0,1}, L_{1,0})
        Si False, usa el parámetro 'angle' como ángulo fijo de rotación.

    angle:
        Ángulo fijo para la transformada rotada cuando use_dominant_orientation=False.
        Por default está en grados, salvo que angle_unit='radians'.

    angle_unit:
        'degrees' o 'radians'.

    compact_rotated:
        Solo se usa si use_rotated=True.
        Si True, devuelve la representación compacta orientada:
            L00, L10^theta, L20^theta, ..., LN0^theta.
        Si False, devuelve todos los coeficientes rotados Lmn^theta.

    include_dc_in_coeffs:
        Si True, incluye L00 en los coeficientes devueltos.

    include_dc_in_energy:
        Si True, la imagen transformada/energía incluye L00.
        Para textura normalmente conviene False, para que no domine la intensidad.

    mode:
        Modo de frontera para scipy.ndimage.gaussian_filter. Ej. 'reflect'.

    scale_normalized:
        Si True, multiplica derivadas por sigma^(m+n) para comparar órdenes.

    channel_axis:
        'last' devuelve coeff_stack como HxWxC.
        'first' devuelve coeff_stack como CxHxW.

    save_image:
        Si True, guarda transformed_image como imagen 8-bit.

    output_path:
        Ruta de guardado. Si None, se crea automáticamente.

    return_dataclass:
        Si True, devuelve HermiteResult; si False, devuelve dict.

    Returns
    -------
    result:
        dict o HermiteResult con:
            transformed_image : np.ndarray HxW float32 en [0,1]
            coeff_stack       : np.ndarray HxWxC o CxHxW float32
            coeffs            : OrderedDict con mapas de coeficientes
            theta             : mapa HxW de orientación o None
            energy            : mapa HxW de energía sin normalizar
            saved_path        : ruta de imagen guardada o None
            params            : parámetros usados
    """
    img = load_image_gray(image)

    # 1) Transformada normal base: coeficientes cartesianos L_(m,n)
    normal_coeffs = normal_hermite_coefficients(
        img,
        max_order=max_order,
        sigma=sigma,
        mode=mode,
        scale_normalized=scale_normalized,
    )

    theta = None

    # 2) Transformada normal o rotada
    if not use_rotated:
        coeffs = normal_coeffs
    else:
        if use_dominant_orientation:
            theta = orientation_from_first_order(normal_coeffs)
        else:
            if angle_unit == "degrees":
                theta_scalar = np.deg2rad(angle)
            elif angle_unit == "radians":
                theta_scalar = float(angle)
            else:
                raise ValueError("angle_unit debe ser 'degrees' o 'radians'.")
            theta = np.full(img.shape, theta_scalar, dtype=np.float32)

        coeffs = rotated_hermite_coefficients(
            normal_coeffs,
            theta=theta,
            max_order=max_order,
            compact=compact_rotated,
            include_dc=include_dc_in_coeffs,
        )

    # 3) Imagen transformada 2D: energía Hermite normalizada
    energy = hermite_energy(coeffs, include_dc=include_dc_in_energy)
    transformed_image = normalize01(energy, robust=True)

    # 4) Stack de coeficientes para modelos
    coeff_stack = coefficients_to_stack(coeffs, channel_axis=channel_axis)

    # 5) Guardado opcional
    saved_path = None
    if save_image:
        if output_path is None:
            suffix = "_hermite_rotada.png" if use_rotated else "_hermite_normal.png"
            output_path = default_output_path(image, suffix=suffix)
        saved_path = save_gray_image(transformed_image, output_path)

    params = {
        "max_order": max_order,
        "sigma": sigma,
        "use_rotated": use_rotated,
        "use_dominant_orientation": use_dominant_orientation,
        "angle": angle,
        "angle_unit": angle_unit,
        "compact_rotated": compact_rotated,
        "include_dc_in_coeffs": include_dc_in_coeffs,
        "include_dc_in_energy": include_dc_in_energy,
        "mode": mode,
        "scale_normalized": scale_normalized,
        "channel_axis": channel_axis,
        "coeff_keys": list(coeffs.keys()),
    }

    result = HermiteResult(
        transformed_image=transformed_image,
        coeff_stack=coeff_stack,
        coeffs=coeffs,
        theta=theta,
        energy=energy,
        saved_path=saved_path,
        params=params,
    )

    return result if return_dataclass else result.to_dict()


def extract_simple_texture_features(
    coeff_stack: Array,
    channel_axis: Literal["last", "first"] = "last",
) -> Array:
    """
    Extrae características simples para clasificación de texturas.

    Para cada canal de coeficientes calcula:
        media, desviación estándar, energía media y valor absoluto medio.

    Returns
    -------
    features:
        Vector 1D np.float32.
    """
    x = np.asarray(coeff_stack, dtype=np.float64)
    if channel_axis == "first":
        x = np.moveaxis(x, 0, -1)  # CxHxW -> HxWxC
    elif channel_axis != "last":
        raise ValueError("channel_axis debe ser 'last' o 'first'.")

    means = x.mean(axis=(0, 1))
    stds = x.std(axis=(0, 1))
    energies = (x * x).mean(axis=(0, 1))
    abs_means = np.abs(x).mean(axis=(0, 1))

    return np.concatenate([means, stds, energies, abs_means]).astype(np.float32)


def demo_with_synthetic_texture() -> dict:
    """
    Demo sin archivos externos: crea una textura sinusoidal diagonal y aplica
    Hermite rotada con orientación dominante.
    """
    h, w = 256, 256
    y, x = np.mgrid[0:h, 0:w]
    img = 0.5 + 0.5 * np.sin(2 * np.pi * (x + y) / 24.0)
    img = img.astype(np.float32)

    result = hermite_transform_image(
        img,
        max_order=3,
        sigma=2.0,
        use_rotated=True,
        use_dominant_orientation=True,
        compact_rotated=True,
        save_image=True,
        output_path="demo_hermite_rotada.png",
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transformada de Hermite normal/rotada para imágenes.")
    parser.add_argument("--image", type=str, default=None, help="Ruta de imagen de entrada.")
    parser.add_argument("--output", type=str, default=None, help="Ruta para guardar imagen transformada.")
    parser.add_argument("--order", type=int, default=3, help="Orden máximo de Hermite.")
    parser.add_argument("--sigma", type=float, default=2.0, help="Escala sigma de la Gaussiana.")
    parser.add_argument("--normal", action="store_true", help="Usar transformada normal en vez de rotada.")
    parser.add_argument("--fixed-angle", action="store_true", help="Usar ángulo fijo en vez de orientación dominante.")
    parser.add_argument("--angle", type=float, default=0.0, help="Ángulo fijo en grados si --fixed-angle está activo.")
    parser.add_argument("--full", action="store_true", help="Devolver todos los coeficientes rotados, no solo compactos.")
    parser.add_argument("--channel-axis", choices=["last", "first"], default="last")
    parser.add_argument("--demo", action="store_true", help="Ejecuta demo con textura sintética.")

    args = parser.parse_args()

    if args.demo:
        out = demo_with_synthetic_texture()
        print("Demo guardado en:", out["saved_path"])
        print("transformed_image shape:", out["transformed_image"].shape)
        print("coeff_stack shape:", out["coeff_stack"].shape)
        print("coeff keys:", out["params"]["coeff_keys"])
    else:
        if args.image is None:
            raise ValueError("Debes pasar --image ruta_imagen o usar --demo.")

        out = hermite_transform_image(
            image=args.image,
            max_order=args.order,
            sigma=args.sigma,
            use_rotated=not args.normal,
            use_dominant_orientation=not args.fixed_angle,
            angle=args.angle,
            angle_unit="degrees",
            compact_rotated=not args.full,
            channel_axis=args.channel_axis,
            save_image=True,
            output_path=args.output,
        )

        print("Imagen transformada guardada en:", out["saved_path"])
        print("transformed_image shape:", out["transformed_image"].shape)
        print("coeff_stack shape:", out["coeff_stack"].shape)
        print("coeficientes:", out["params"]["coeff_keys"])
