"""Ejemplo mínimo de uso de la implementación modular."""

from pathlib import Path

from hermite_transform import hermite_transform_image


HERE = Path(__file__).resolve().parent
HERMITE_DIRECTORY = HERE.parent

demo_image = HERMITE_DIRECTORY / "Fusion_Images_ds" / "house.tif"
# demo_image = HERMITE_DIRECTORY / "Fusion_Images_ds" / "lena.jpg"

result = hermite_transform_image(
    image=demo_image,
    max_order=3,
    sigma=2.0,
    coefficient_region="square",
    sampling_step=1,
    boundary="symmetric",
    use_rotation=True,
    use_inverse_rotation=True,
    use_inverse_transform=True,
    rotation_mode="dominant",
    results_path=HERE / "results",
)

print("Órdenes visibles:", result["orders"])
print("Radio del soporte:", result["support_radius"])
print("Métricas de steering:", result["coefficient_roundtrip_metrics"])
print("Métricas de reconstrucción:", result["reconstruction_metrics"])
print("Resultados guardados en:", result["results_path"])
