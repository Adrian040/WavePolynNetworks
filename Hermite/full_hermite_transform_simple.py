"""Transformada de Hermite 2-D con filtros Hermite--Gaussianos.

Versión simplificada para estudiar el flujo matemático:
imagen -> filtros -> coeficientes -> steering -> inverse steering -> síntesis.
"""

from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import correlate1d, convolve1d
from scipy.special import eval_hermite, gammaln


# -----------------------------------------------------------------------------
# 1. Imagen
# -----------------------------------------------------------------------------


def read_image(image):
    """Lee la imagen y la deja como una matriz 2-D float64 en grayscale."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            # Esto basta para los formatos comunes de prueba.
            return np.asarray(im.convert("L"), dtype=np.float64)

    arr = np.asarray(image)

    if arr.ndim == 2:
        return arr.astype(np.float64)

    if arr.ndim == 3:
        # Para 1 o 2 canales: usamos solo el canal de intensidad (el primero).
        if arr.shape[2] <= 2:
            return arr[..., 0].astype(np.float64)

        # RGB/RGBA: ignoramos alpha y convertimos a luminancia.
        rgb = arr[..., :3].astype(np.float64)
        return (
            0.298936 * rgb[..., 0]
            + 0.587043 * rgb[..., 1]
            + 0.114021 * rgb[..., 2]
        )

    raise ValueError("La imagen debe ser grayscale, RGB o RGBA.")


# -----------------------------------------------------------------------------
# 2. Órdenes L_mn
# -----------------------------------------------------------------------------


def hermite_orders(max_order, region="triangle"):
    """Orden L00, L10, L01, L20, L11, L02, ..."""
    orders = []

    if region == "triangle":
        for total in range(max_order + 1):
            for m in range(total, -1, -1):
                orders.append((m, total - m))

    elif region == "square":
        for total in range(2 * max_order + 1):
            for m in range(min(max_order, total), -1, -1):
                n = total - m
                if n <= max_order:
                    orders.append((m, n))

    else:
        raise ValueError("region debe ser 'triangle' o 'square'.")

    return orders


# -----------------------------------------------------------------------------
# 3. Filtros Hermite--Gaussianos
# -----------------------------------------------------------------------------


def hermite_filter_1d(n, x, sigma):
    """a_n(x) = H_n(x/sigma)/sqrt(2^n n!) * Gaussiana."""
    u = x / sigma

    norm = np.exp(-0.5 * (n * np.log(2.0) + gammaln(n + 1)))
    Hn = norm * eval_hermite(n, u)
    gaussian = np.exp(-(u**2)) / (sigma * np.sqrt(np.pi))

    return Hn * gaussian


def choose_support(sigma, highest_order, tol=1e-8):
    """Aumenta el soporte hasta que las colas de todos los filtros sean pequeñas."""
    radius = int(
        np.ceil(sigma * max(4.0, np.sqrt(2 * highest_order + 1) + 2.0))
    )

    while True:
        x = np.arange(radius + 3, dtype=float)
        good = True

        for n in range(highest_order + 1):
            h = np.abs(hermite_filter_1d(n, x, sigma))
            if np.any(h[radius:] / np.max(h) >= tol):
                good = False
                break

        if good:
            return radius

        radius += 1


def build_filter_bank(max_order=3, sigma=2.0, region="square"):
    """Construye los filtros 1-D necesarios."""
    visible_orders = hermite_orders(max_order, region)

    # El square necesita órdenes auxiliares para tener bloques completos
    # durante el steering.
    if region == "square":
        steering_orders = hermite_orders(2 * max_order, "triangle")
    else:
        steering_orders = visible_orders.copy()

    highest_order = max(max(m, n) for m, n in steering_orders)
    radius = choose_support(sigma, highest_order)
    x = np.arange(-radius, radius + 1, dtype=float)

    filters = {
        n: hermite_filter_1d(n, x, sigma)
        for n in range(highest_order + 1)
    }

    return {
        "orders": visible_orders,
        "steering_orders": steering_orders,
        "filters": filters,
        "radius": radius,
        "sigma": sigma,
        "max_order": max_order,
        "region": region,
    }


# -----------------------------------------------------------------------------
# 4. Transformada cartesiana
# -----------------------------------------------------------------------------


def cartesian_transform(image, bank, orders=None, sampling_step=1):
    """Calcula L_mn con filtros separables."""
    if orders is None:
        orders = bank["orders"]

    coeffs = {}
    horizontal = {}

    # m actúa en x -> columnas.
    for m, _ in orders:
        if m not in horizontal:
            horizontal[m] = correlate1d(
                image,
                bank["filters"][m],
                axis=1,
                mode="reflect",
            )

    # n actúa en y -> filas.
    for m, n in orders:
        response = correlate1d(
            horizontal[m],
            bank["filters"][n],
            axis=0,
            mode="reflect",
        )

        coeffs[(m, n)] = response[::sampling_step, ::sampling_step]

    return coeffs


# -----------------------------------------------------------------------------
# 5. Steering
# -----------------------------------------------------------------------------


def dominant_theta(coeffs):
    """theta = atan2(L01, L10)."""
    return np.arctan2(coeffs[(0, 1)], coeffs[(1, 0)])


def rotate_block(block, theta):
    """Rota un bloque [L_r0, L_(r-1)1, ..., L_0r]."""
    degree = block.shape[-1] - 1

    if degree == 0:
        return block.copy()

    c = np.cos(theta)
    s = np.sin(theta)

    # Raíces de la fila correspondiente del triángulo de Pascal.
    pascal = np.sqrt([comb(degree, k) for k in range(degree + 1)])

    work = block.astype(float).copy()
    if degree > 1:
        work[..., 1:degree] /= pascal[1:degree]

    rotated = np.empty_like(work)
    active = degree + 1

    for out in range(degree):
        reduced = work.copy()
        length = active

        # Repetimos c*x + s*y las veces necesarias.
        for _ in range(out, degree):
            reduced = (
                c[..., None] * reduced[..., : length - 1]
                + s[..., None] * reduced[..., 1:length]
            )
            length -= 1

        rotated[..., out] = reduced[..., 0] * pascal[out]

        # Dirección perpendicular: c*y - s*x.
        work = (
            c[..., None] * work[..., 1:active]
            - s[..., None] * work[..., : active - 1]
        )
        active -= 1

    rotated[..., degree] = work[..., 0]
    return rotated


def rotate_coefficients(coeffs, theta, orders):
    """Rota cada nivel de orden total por separado."""
    result = {order: coeffs[order].copy() for order in orders}

    for total in sorted({m + n for m, n in orders}):
        block_orders = [(m, total - m) for m in range(total, -1, -1)]

        if not all(order in coeffs for order in block_orders):
            raise ValueError(f"El bloque de orden {total} no está completo.")

        block = np.stack([coeffs[o] for o in block_orders], axis=-1)
        rotated = rotate_block(block, theta)

        for k, order in enumerate(block_orders):
            result[order] = rotated[..., k]

    return result


def inverse_rotate_coefficients(coeffs, theta, orders):
    """Deshace el steering usando -theta."""
    return rotate_coefficients(coeffs, -theta, orders)


# -----------------------------------------------------------------------------
# 6. Síntesis
# -----------------------------------------------------------------------------


def synthesize(coeffs, bank, image_shape, sampling_step=1):
    """Reconstrucción truncada mediante overlap-add y normalización por w^2."""
    h, w = image_shape
    rows = np.arange(0, h, sampling_step)
    cols = np.arange(0, w, sampling_step)

    numerator = np.zeros((h, w), dtype=float)

    # Sumamos L_mn * a_mn desplazadas a cada posición de análisis.
    for m, n in bank["orders"]:
        up = np.zeros((h, w), dtype=float)
        up[np.ix_(rows, cols)] = coeffs[(m, n)]

        temp = convolve1d(
            up,
            bank["filters"][m],
            axis=1,
            mode="constant",
        )

        numerator += convolve1d(
            temp,
            bank["filters"][n],
            axis=0,
            mode="constant",
        )

    # Denominador = suma de las ventanas w^2 desplazadas.
    mask = np.zeros((h, w), dtype=float)
    mask[np.ix_(rows, cols)] = 1.0

    denominator = convolve1d(
        mask,
        bank["filters"][0],
        axis=1,
        mode="constant",
    )
    denominator = convolve1d(
        denominator,
        bank["filters"][0],
        axis=0,
        mode="constant",
    )

    return numerator / denominator


# -----------------------------------------------------------------------------
# 7. Visualización mínima
# -----------------------------------------------------------------------------


def show_coefficients(coeffs, title="Coeficientes"):
    """Muestra los mapas L_mn en su posición (m,n)."""
    max_m = max(m for m, _ in coeffs)
    max_n = max(n for _, n in coeffs)

    fig, axes = plt.subplots(
        max_n + 1,
        max_m + 1,
        figsize=(2.3 * (max_m + 1), 2.3 * (max_n + 1)),
        squeeze=False,
    )

    for n in range(max_n + 1):
        for m in range(max_m + 1):
            ax = axes[n, m]

            if (m, n) in coeffs:
                value = coeffs[(m, n)]
                limit = max(np.max(np.abs(value)), 1e-12)
                ax.imshow(value, cmap="gray", vmin=-limit, vmax=limit)
                ax.set_title(f"L{m}{n}")

            ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 8. Flujo principal
# -----------------------------------------------------------------------------


def hermite_transform_image(
    image,
    max_order=3,
    sigma=2.0,
    coefficient_region="square",
    sampling_step=1,
    use_rotation=True,
    use_inverse_rotation=True,
    use_inverse_transform=True,
    rotation_mode="dominant",
    angle=0.0,
    angle_unit="degrees",
):
    """Transformada -> steering -> inverse steering -> síntesis."""
    image = read_image(image)

    bank = build_filter_bank(
        max_order=max_order,
        sigma=sigma,
        region=coefficient_region,
    )

    visible_orders = bank["orders"]

    # Para square calculamos los auxiliares requeridos por steering.
    analysis_orders = bank["steering_orders"] if use_rotation else visible_orders

    all_cartesian = cartesian_transform(
        image,
        bank,
        orders=analysis_orders,
        sampling_step=sampling_step,
    )

    cartesian = {o: all_cartesian[o] for o in visible_orders}

    theta = None
    rotated = None
    recovered = None

    if use_rotation:
        if rotation_mode in ("dominant", "gradient", "grad"):
            theta = dominant_theta(all_cartesian)
        else:
            theta = np.asarray(angle, dtype=float)
            if angle_unit.startswith("deg"):
                theta = np.deg2rad(theta)

        all_rotated = rotate_coefficients(
            all_cartesian,
            theta,
            bank["steering_orders"],
        )
        rotated = {o: all_rotated[o] for o in visible_orders}

        if use_inverse_rotation or use_inverse_transform:
            all_recovered = inverse_rotate_coefficients(
                all_rotated,
                theta,
                bank["steering_orders"],
            )
            recovered = {o: all_recovered[o] for o in visible_orders}

    reconstructed = None

    if use_inverse_transform:
        coeffs_for_synthesis = recovered if use_rotation else cartesian
        reconstructed = synthesize(
            coeffs_for_synthesis,
            bank,
            image.shape,
            sampling_step=sampling_step,
        )

    return {
        "original_image": image,
        "cartesian_coefficients": cartesian,
        "rotated_coefficients": rotated,
        "recovered_cartesian_coefficients": recovered,
        "theta": theta,
        "reconstructed_image": reconstructed,
        "orders": visible_orders,
        "auxiliary_orders": bank["steering_orders"],
        "filter_bank": bank,
    }


if __name__ == "__main__":
    result = hermite_transform_image(
        image=Path("Fusion_Images_ds") / "house.tif",
        max_order=3,
        sigma=2.0,
        coefficient_region="square",
    )

    print("Órdenes:", result["orders"])
    print("Radio del soporte:", result["filter_bank"]["radius"])

    show_coefficients(
        result["cartesian_coefficients"],
        "Coeficientes cartesianos",
    )

    if result["rotated_coefficients"] is not None:
        show_coefficients(
            result["rotated_coefficients"],
            "Coeficientes rotados",
        )

    if result["reconstructed_image"] is not None:
        plt.figure(figsize=(6, 5))
        plt.imshow(result["reconstructed_image"], cmap="gray")
        plt.title("Reconstrucción")
        plt.axis("off")
        plt.show()
