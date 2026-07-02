import os
from math import comb, factorial

import numpy as np
from PIL import Image
from scipy.special import eval_hermite
from scipy.signal import correlate2d, convolve2d
import matplotlib.pyplot as plt


# -------------------------
# Utilidades básicas
# -------------------------

def read_image(image):
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
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def hermite_orders(max_order):
    # (m,n): m = orden en x, n = orden en y, con m+n <= max_order
    return [(m, total - m) for total in range(max_order + 1) for m in range(total + 1)]


# -------------------------
# Base Hermite del paper
# -------------------------

def hermite_basis(max_order=3, sigma=2.0, kernel_size=None, discrete_normalize=True):
    """Crea G_mn(x,y) y la ventana gaussiana w(x,y)."""
    if kernel_size is None:
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    r = kernel_size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    xs = x / sigma
    ys = y / sigma

    # Ventana gaussiana sin constante global. La constante global se cancela en la normalización.
    w = np.exp(-(xs**2 + ys**2) / 2).astype(np.float32)
    w2 = w**2

    basis = {}
    for m, n in hermite_orders(max_order):
        total = m + n

        # Constante de normalización del paper para los polinomios Hermite:
        # 1 / sqrt(2^(m+n) m! n!)
        const = 1.0 / np.sqrt((2.0 ** total) * factorial(m) * factorial(n))

        G = const * eval_hermite(m, xs) * eval_hermite(n, ys)
        G = G.astype(np.float32)

        # Corrección discreta opcional: hace que sum w^2 G^2 ≈ 1 en la ventana discreta.
        # Esto mejora la reconstrucción numérica en imágenes digitales.
        if discrete_normalize:
            norm = np.sqrt(np.sum(w2 * G * G))
            G = G / (norm + 1e-8)

        basis[(m, n)] = G.astype(np.float32)

    return basis, w.astype(np.float32), w2.astype(np.float32)


# -------------------------
# Transformada directa e inversa
# -------------------------

def hermite_coefficients(
    image,
    max_order=3,
    sigma=2.0,
    kernel_size=None,
    boundary="symm",
    discrete_normalize=True,
):
    """Transformada de Hermite directa: L_mn = I * D_mn, con D_mn = G_mn w^2."""
    img = read_image(image)
    basis, w, w2 = hermite_basis(max_order, sigma, kernel_size, discrete_normalize)

    coeffs = {}
    for k, G in basis.items():
        D = G * w2
        coeffs[k] = correlate2d(img, D, mode="same", boundary=boundary).astype(np.float32)

    return coeffs, basis, w


def inverse_hermite_transform(coeffs, basis, w, image_shape, boundary="fill"):
    """Reconstrucción por síntesis/overlap-add usando los coeficientes Hermite."""
    numerator = np.zeros(image_shape, dtype=np.float32)

    for k, c_map in coeffs.items():
        S = basis[k] * w
        numerator += convolve2d(c_map, S, mode="same", boundary=boundary, fillvalue=0).astype(np.float32)

    # Normalización W(x,y): suma de ventanas sobre la malla. Aquí se usa malla densa pixel a pixel.
    W = convolve2d(
        np.ones(image_shape, dtype=np.float32),
        w,
        mode="same",
        boundary=boundary,
        fillvalue=0,
    ).astype(np.float32)

    recon = numerator / (W + 1e-8)
    return recon.astype(np.float32)


# -------------------------
# Rotación / steering
# -------------------------

def dominant_theta(coeffs):
    """theta = arctan2(L_01, L_10)."""
    if (1, 0) not in coeffs or (0, 1) not in coeffs:
        return np.zeros_like(next(iter(coeffs.values())), dtype=np.float32)
    theta = np.arctan2(coeffs[(0, 1)], coeffs[(1, 0)] + 1e-8)
    return np.mod(theta, np.pi).astype(np.float32)


def rotate_coefficients(coeffs, theta, max_order):
    """Rota todos los coeficientes con Dx'=cos(theta)Dx+sin(theta)Dy."""
    c = np.cos(theta)
    s = np.sin(theta)
    rotated = {}
    shape = next(iter(coeffs.values())).shape

    for a, b in hermite_orders(max_order):
        out = np.zeros(shape, dtype=np.float32)
        for i in range(a + 1):
            coef1 = comb(a, i) * (c ** (a - i)) * (s ** i)
            for j in range(b + 1):
                coef2 = comb(b, j) * ((-s) ** (b - j)) * (c ** j)
                ox = (a - i) + (b - j)
                oy = i + j
                out += coef1 * coef2 * coeffs[(ox, oy)]
        rotated[(a, b)] = out.astype(np.float32)

    return rotated


# -------------------------
# Salidas para visualizar/modelos
# -------------------------

def coeffs_to_stack(coeffs, order_list=None, channel_axis="last"):
    if order_list is None:
        order_list = list(coeffs.keys())
    stack = np.stack([coeffs[k] for k in order_list], axis=-1).astype(np.float32)
    if channel_axis == "first":
        stack = np.moveaxis(stack, -1, 0)
    return stack


def energy_image(coeffs, include_dc=False):
    e = np.zeros_like(next(iter(coeffs.values())), dtype=np.float32)
    for k, v in coeffs.items():
        if not include_dc and k == (0, 0):
            continue
        e += v**2
    return normalize01(np.sqrt(e))


def save_coefficients_grid(coeffs, output_path, title="Coeficientes Hermite"):
    orders = list(coeffs.keys())
    max_m = max(m for m, _ in orders)
    max_n = max(n for _, n in orders)
    cols = max_m + 1
    rows = max_n + 1

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for n in range(rows):
        for m in range(cols):
            ax = axes[n, m]
            k = (m, n)
            if k in coeffs:
                ax.imshow(normalize01(coeffs[k]), cmap="gray")
                ax.set_title(rf"$L_{{{m},{n}}}$")
            ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_reconstruction_comparison(original, reconstructed, output_path):
    error = np.abs(original - reconstructed)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="gray", vmin=0, vmax=1)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(normalize01(reconstructed), cmap="gray")
    plt.title("Reconstruida")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(normalize01(error), cmap="gray")
    plt.title("Error abs.")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def reconstruction_metrics(original, reconstructed):
    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)
    mse = np.mean((original - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(original - reconstructed))
    psnr = 20 * np.log10(1.0 / (rmse + 1e-12))
    return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "psnr": float(psnr)}


# -------------------------
# Función principal
# -------------------------

def hermite_transform_image(
    image,
    max_order=3,
    sigma=2.0,
    kernel_size=None,
    use_rotated=True,
    use_dominant_orientation=True,
    angle=0.0,
    angle_unit="degrees",
    save_image=True,
    output_path="hermite_coefficients.png",
    save_reconstruction=True,
    reconstruction_path="hermite_reconstruction.png",
    channel_axis="last",
    discrete_normalize=True,
):
    img = read_image(image)

    # Transformada normal corregida: usa G_mn con constante del paper y D_mn = G_mn w^2.
    normal_coeffs, basis, w = hermite_coefficients(
        img,
        max_order=max_order,
        sigma=sigma,
        kernel_size=kernel_size,
        discrete_normalize=discrete_normalize,
    )

    # Inversa de la transformada normal. Esta es la prueba de consistencia de la HT.
    reconstructed = inverse_hermite_transform(normal_coeffs, basis, w, img.shape)
    metrics = reconstruction_metrics(img, np.clip(reconstructed, 0, 1))

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

    if save_reconstruction:
        save_reconstruction_comparison(img, np.clip(reconstructed, 0, 1), reconstruction_path)

    return {
        "transformed_image": transformed_image,
        "coeff_stack": coeff_stack,
        "coeffs": coeffs,
        "normal_coeffs": normal_coeffs,
        "orders": orders,
        "theta": theta,
        "reconstructed_image": np.clip(reconstructed, 0, 1).astype(np.float32),
        "reconstruction_metrics": metrics,
        "basis": basis,
        "window": w,
        "output_path": output_path if save_image else None,
        "reconstruction_path": reconstruction_path if save_reconstruction else None,
    }


if __name__ == "__main__":
    result = hermite_transform_image(
        image="mi_imagen.png",
        max_order=5,
        sigma=2.0,
        use_rotated=True,
        use_dominant_orientation=True,
        save_image=True,
        output_path="coeficientes_hermite_rotados.png",
        save_reconstruction=True,
        reconstruction_path="reconstruccion_hermite.png",
    )
    print("Imagen transformada:", result["transformed_image"].shape)
    print("Stack de coeficientes:", result["coeff_stack"].shape)
    print("Órdenes:", result["orders"])
    print("Métricas reconstrucción:", result["reconstruction_metrics"])
