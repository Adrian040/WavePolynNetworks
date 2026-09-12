"""Orquestación del análisis, steering, síntesis y guardado de resultados."""

from pathlib import Path

import numpy as np
from PIL import Image

from .analysis import cartesian_transform, normalize_boundary
from .filters import build_filter_bank
from .image_io import read_image
from .metrics import (
    coefficient_energy,
    coefficient_roundtrip_metrics,
    coefficients_to_stack,
    reconstruction_metrics,
)
from .steering import (
    angle_map,
    dominant_theta,
    inverse_rotate_coefficients,
    rotate_coefficients,
)
from .synthesis import synthesize
from .visualization import (
    save_coefficients_grid,
    save_intensity_image,
    save_metrics_csv,
    save_reconstruction_comparison,
    save_theta_image,
)


Array = np.ndarray
Order = tuple[int, int]


def _fixed_angle(angle: float | Array, angle_unit: str) -> Array:
    """Convierte un ángulo fijo a radianes."""
    if not isinstance(angle_unit, str):
        raise ValueError("angle_unit debe ser 'degrees' o 'radians'.")
    unit = angle_unit.lower()
    if unit in {"degree", "degrees", "deg"}:
        return np.deg2rad(np.asarray(angle, dtype=np.float64))
    if unit in {"radian", "radians", "rad"}:
        return np.asarray(angle, dtype=np.float64)
    raise ValueError("angle_unit debe ser 'degrees' o 'radians'.")


def hermite_transform_image(
    image: str | Path | Image.Image | Array,
    max_order: int = 3,
    sigma: float = 2.0,
    coefficient_region: str = "square",
    sampling_step: int = 1,
    boundary: str = "symmetric",
    use_rotation: bool = True,
    use_inverse_rotation: bool = True,
    use_inverse_transform: bool = True,
    rotation_mode: str = "dominant",
    angle: float | Array = 0.0,
    angle_unit: str = "degrees",
    results_path: str | Path = "results",
) -> dict:
    """Ejecuta el flujo completo y guarda todos los resultados disponibles.

    Parameters
    ----------
    image : str, Path, PIL.Image.Image or ndarray
        Imagen grayscale, RGB o RGBA.
    max_order : int, default=3
        Límite total para ``triangle`` o límite por eje para ``square``.
    sigma : float, default=2.0
        Escala Gaussiana en píxeles.
    coefficient_region : {"triangle", "square"}, default="square"
        Región pública de coeficientes.
    sampling_step : int, default=1
        Separación de la retícula local.
    boundary : {"symmetric", "edge", "constant", "wrap"}
        Extensión de la imagen durante el análisis.
    use_rotation, use_inverse_rotation, use_inverse_transform : bool
        Activa steering, su inversión y la síntesis, respectivamente.
    rotation_mode : {"dominant", "fixed"}, default="dominant"
        Fuente del ángulo de steering.
    angle : float or ndarray, default=0.0
        Ángulo escalar o mapa para el modo fijo.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Unidad del ángulo fijo.
    results_path : str or Path, default="results"
        Carpeta única de salida; se crea automáticamente.

    Returns
    -------
    dict
        Coeficientes, orientación, reconstrucción, métricas, banco de filtros,
        energía, stack y rutas de todos los archivos guardados.
    """
    if use_inverse_rotation and not use_rotation:
        raise ValueError("use_inverse_rotation=True requiere use_rotation=True.")
    boundary, _ = normalize_boundary(boundary)
    image = read_image(image)
    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    saved_files: dict[str, Path] = {}

    def save(key: str, filename: str, function, *args) -> None:
        """Guarda una salida y registra su ruta en el resultado."""
        path = results_path / filename
        function(*args, path)
        saved_files[key] = path

    save(
        "original_image",
        "01_original_input_grayscale_image.png",
        save_intensity_image,
        image,
        "Imagen original en escala de grises",
    )

    bank = build_filter_bank(max_order, sigma, coefficient_region)
    visible_orders = list(bank["orders"])
    steering_orders = list(bank["steering_orders"])
    analysis_orders = steering_orders if use_rotation else visible_orders
    full_cartesian = cartesian_transform(
        image,
        bank,
        analysis_orders,
        sampling_step,
        boundary,
    )
    cartesian = {order: full_cartesian[order].copy() for order in visible_orders}
    save(
        "cartesian_coefficients",
        "02_cartesian_hermite_coefficient_maps.png",
        save_coefficients_grid,
        cartesian,
        "Coeficientes Hermite cartesianos",
    )

    theta = None
    rotated = None
    recovered = None
    coefficient_metrics = None
    if use_rotation:
        if not isinstance(rotation_mode, str):
            raise ValueError("rotation_mode debe ser 'dominant' o 'fixed'.")
        mode = rotation_mode.lower()
        if mode in {"dominant", "gradient", "grad"}:
            theta = dominant_theta(full_cartesian)
        elif mode in {"fixed", "angle"}:
            theta = _fixed_angle(angle, angle_unit)
        else:
            raise ValueError("rotation_mode debe ser 'dominant' o 'fixed'.")

        full_rotated = rotate_coefficients(full_cartesian, theta, steering_orders)
        rotated = {order: full_rotated[order].copy() for order in visible_orders}
        save(
            "rotated_coefficients",
            "03_steered_rotated_hermite_coefficient_maps.png",
            save_coefficients_grid,
            rotated,
            "Coeficientes Hermite rotados por steering",
        )
        save(
            "theta",
            "04_local_gradient_orientation_theta_in_degrees.png",
            save_theta_image,
            angle_map(theta, next(iter(cartesian.values())).shape),
        )

        if use_inverse_rotation or use_inverse_transform:
            full_recovered = inverse_rotate_coefficients(
                full_rotated,
                theta,
                steering_orders,
            )
            recovered = {
                order: full_recovered[order].copy()
                for order in visible_orders
            }
            coefficient_metrics = coefficient_roundtrip_metrics(
                cartesian,
                recovered,
            )
            save(
                "recovered_cartesian_coefficients",
                "05_recovered_cartesian_from_rotated_coefficients.png",
                save_coefficients_grid,
                recovered,
                "Coeficientes cartesianos recuperados desde los rotados",
            )

    reconstructed = None
    image_metrics = None
    if use_inverse_transform:
        synthesis_coefficients = recovered if use_rotation else cartesian
        reconstructed = synthesize(
            synthesis_coefficients,
            bank,
            image.shape,
            sampling_step,
        )
        image_metrics = reconstruction_metrics(image, reconstructed)
        save(
            "reconstructed_image",
            "06_truncated_hermite_synthesis_reconstructed_image.png",
            save_intensity_image,
            reconstructed,
            "Imagen reconstruida por síntesis Hermite truncada",
        )
        save(
            "reconstruction_comparison",
            "07_reconstruction_comparison_with_absolute_error.png",
            save_reconstruction_comparison,
            image,
            reconstructed,
        )

    active_coefficients = rotated if use_rotation else cartesian
    stack = coefficients_to_stack(active_coefficients, visible_orders)
    energy = coefficient_energy(active_coefficients)
    energy_kind = "rotated" if use_rotation else "cartesian"
    save(
        "coefficient_energy",
        f"08_{energy_kind}_hermite_coefficient_energy_without_dc.png",
        save_intensity_image,
        energy,
        "Energía de coeficientes Hermite sin componente DC",
    )

    metrics = {}
    if coefficient_metrics:
        metrics.update(coefficient_metrics)
    if image_metrics:
        metrics.update(image_metrics)
    if metrics:
        save(
            "metrics_csv",
            "09_steering_roundtrip_and_reconstruction_metrics.csv",
            save_metrics_csv,
            metrics,
        )

    return {
        "original_image": image,
        "cartesian_coefficients": cartesian,
        "rotated_coefficients": rotated,
        "recovered_cartesian_coefficients": recovered,
        "theta": theta,
        "reconstructed_image": reconstructed,
        "coefficient_roundtrip_metrics": coefficient_metrics,
        "reconstruction_metrics": image_metrics,
        "active_coefficients": active_coefficients,
        "coeff_stack": stack,
        "coefficient_energy": energy,
        "transformed_image": energy,
        "orders": visible_orders,
        "auxiliary_orders": steering_orders,
        "filter_bank": bank,
        "support_radius": bank["radius"],
        "boundary": boundary,
        "results_path": results_path,
        "saved_files": saved_files,
    }
