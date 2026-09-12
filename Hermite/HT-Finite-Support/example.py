"""Ejemplo mínimo de uso de la implementación modular."""

from pathlib import Path

from hermite_transform import hermite_transform_image


HERE = Path(__file__).resolve().parent
print(HERE)
EXAMPLE_IMAGES = HERE / "example_images"

# demo_image = EXAMPLE_IMAGES / "house.tif"
demo_image = EXAMPLE_IMAGES / "lena.jpg"

result = hermite_transform_image(
    image=demo_image,
    max_order=3,
    sigma=2.0,
    coefficient_region="triangle",
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

print("==== Ejemplos para obtener coefs. individuales ===")
# Ejemplos para el coef. (0,1), theta y la imagen reconstruida.
L01 = result["cartesian_coefficients"][(0, 1)]
# L01 = result["rotated_coefficients"][(0, 1)]
# L01 = result["recovered_cartesian_coefficients"][(0, 1)]
# theta = result["theta"]
# reconst_image = result["reconstructed_image"]
# print(theta.shape)
# print(theta)
# print(reconst_image.shape)
# print(reconst_image)
print(L01.shape)
print(L01)

# Visualizar:
import matplotlib.pyplot as plt
plt.imshow(L01, cmap="gray")
# plt.imshow(theta, cmap="gray")
# plt.imshow(reconst_image, cmap="gray")
# plt.title("Coeficiente cartesiano L01")
plt.colorbar()
plt.show()