import os
from math import comb

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def read_image(image):
    """Acepta ruta, PIL.Image o np.ndarray y devuelve imagen gris float32 en [0,1]."""
    if isinstance(image, str):
        img = Image.open(image).convert("L")
        img = np.array(img)
    elif isinstance(image, Image.Image):
        img = np.array(image.convert("L"))
    else:
        img = np.asarray(image)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)

    img = img.astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0
    return img


def normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


def hermite_orders(max_order):
    """Órdenes (m,n) con m+n <= max_order."""
    orders = []
    for total in range(max_order + 1):
        for m in range(total + 1):
            n = total - m
            orders.append((m, n))
    return orders


def hermite_coefficients(image, max_order=3, sigma=2.0, mode="reflect"):
    """Coeficientes Hermite normales usando derivadas Gaussianas."""
    img = read_image(image)
    coeffs = {}

    for m, n in hermite_orders(max_order):
        c = gaussian_filter(img, sigma=sigma, order=(m, n), mode=mode)
        c = c * (sigma ** (m + n))  # normalización por escala
        coeffs[(m, n)] = c.astype(np.float32)

    return coeffs


def dominant_theta(coeffs):
    """Ángulo local theta = arctan2(L_01, L_10)."""
    if (1, 0) not in coeffs or (0, 1) not in coeffs:
        return np.zeros_like(next(iter(coeffs.values())), dtype=np.float32)
    theta = np.arctan2(coeffs[(0, 1)], coeffs[(1, 0)] + 1e-8)
    return theta.astype(np.float32)


def rotate_coefficients(coeffs, theta, max_order):
    """
    Rota todos los coeficientes Hermite.
    Usa Dx' = cos(theta)Dx + sin(theta)Dy,
        Dy' = -sin(theta)Dx + cos(theta)Dy.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    rotated = {}

    for a, b in hermite_orders(max_order):
        out = np.zeros_like(next(iter(coeffs.values())), dtype=np.float32)

        for i in range(a + 1):
            coef1 = comb(a, i) * (c ** (a - i)) * (s ** i)
            for j in range(b + 1):
                coef2 = comb(b, j) * ((-s) ** (b - j)) * (c ** j)
                ox = (a - i) + (b - j)
                oy = i + j
                out += coef1 * coef2 * coeffs[(ox, oy)]

        rotated[(a, b)] = out.astype(np.float32)

    return rotated


def coeffs_to_stack(coeffs, order_list=None, channel_axis="last"):
    """Convierte el diccionario de coeficientes a arreglo HxWxC o CxHxW."""
    if order_list is None:
        order_list = list(coeffs.keys())

    stack = np.stack([coeffs[k] for k in order_list], axis=-1).astype(np.float32)
    if channel_axis == "first":
        stack = np.moveaxis(stack, -1, 0)
    return stack


def energy_image(coeffs, include_dc=False):
    """Imagen transformada resumida como energía de coeficientes."""
    e = np.zeros_like(next(iter(coeffs.values())), dtype=np.float32)
    for k, v in coeffs.items():
        if not include_dc and k == (0, 0):
            continue
        e += v ** 2
    return normalize01(np.sqrt(e))


def save_coefficients_grid(coeffs, output_path, title="Coeficientes Hermite"):
    """Guarda una figura con todos los mapas de coeficientes."""
    orders = list(coeffs.keys())
    n = len(orders)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(3 * cols, 3 * rows))
    for idx, k in enumerate(orders):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(normalize01(coeffs[k]), cmap="gray")
        plt.title(f"L{k[0]},{k[1]}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def hermite_transform_image(
    image,
    max_order=3,
    sigma=2.0,
    use_rotated=True,
    use_dominant_orientation=True,
    angle=0.0,
    angle_unit="degrees",
    save_image=True,
    output_path="hermite_coefficients.png",
    channel_axis="last",
):
    """
    Función principal.
    Devuelve imagen transformada, coeficientes, stack para modelos y theta.
    """
    img = read_image(image)
    normal_coeffs = hermite_coefficients(img, max_order=max_order, sigma=sigma)

    theta = None
    if use_rotated:
        if use_dominant_orientation:
            theta = dominant_theta(normal_coeffs)
        else:
            theta_value = np.deg2rad(angle) if angle_unit == "degrees" else angle
            theta = np.full_like(img, theta_value, dtype=np.float32)
        coeffs = rotate_coefficients(normal_coeffs, theta, max_order=max_order)
        title = "Coeficientes Hermite rotados"
    else:
        coeffs = normal_coeffs
        title = "Coeficientes Hermite normales"

    orders = hermite_orders(max_order)
    transformed_image = energy_image(coeffs)
    coeff_stack = coeffs_to_stack(coeffs, orders, channel_axis=channel_axis)

    if save_image:
        save_coefficients_grid(coeffs, output_path, title=title)

    return {
        "transformed_image": transformed_image,  # HxW, útil como imagen resumen
        "coeff_stack": coeff_stack,             # HxWxC o CxHxW, útil para modelos
        "coeffs": coeffs,                       # diccionario {(m,n): mapa}
        "orders": orders,                       # orden de canales del stack
        "theta": theta,                         # mapa de orientación si se rotó
        "output_path": output_path if save_image else None,
    }


if __name__ == "__main__":
    result = hermite_transform_image(
        image="mi_imagen.png",
        max_order=3,
        sigma=2.0,
        use_rotated=True,
        use_dominant_orientation=True,
        save_image=True,
        output_path="coeficientes_hermite_rotados.png",
    )
    print("Imagen transformada:", result["transformed_image"].shape)
    print("Stack de coeficientes:", result["coeff_stack"].shape)
    print("Órdenes:", result["orders"])
