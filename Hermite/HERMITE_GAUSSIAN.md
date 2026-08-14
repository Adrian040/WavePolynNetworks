# Transformada de Hermite--Gauss de soporte finito

`full_hermite_transform_refactored.py` implementa una descomposición local
bidimensional con ventana Gaussiana y polinomios de Hermite de físicos
normalizados. Esta realización es independiente del banco binomial/Krawtchouk
de `dhts_py`.

- `max_order` limita el orden total en `triangle` y el orden de cada eje en
  `square`.
- `sigma` es la escala Gaussiana expresada en píxeles. El soporte impar y
  simétrico se selecciona automáticamente comprobando las colas de todos los
  filtros necesarios.
- `triangle` conserva los pares con `m+n <= max_order`; `square` conserva
  `0 <= m,n <= max_order`.
- `sampling_step` es la distancia entre posiciones de análisis. Un valor de 1
  produce mapas densos; valores mayores producen mapas submuestreados.
- El flujo por defecto es análisis cartesiano, estimación de orientación,
  steering, steering inverso y síntesis.

Una región `square` no contiene por sí sola bloques rotacionales completos por
encima de `max_order`. Por ello, durante el steering se calcula internamente el
triángulo completo hasta orden total `2*max_order`; las salidas visibles siguen
conteniendo solamente el cuadrado solicitado.

La síntesis coloca los coeficientes en la retícula de análisis, suma las
funciones Hermite--Gaussianas desplazadas y divide por la suma de las ventanas
Gaussianas al cuadrado. Como se conserva un número finito de órdenes, el
resultado es una aproximación truncada y las métricas reportan su error real.

## Lectura y visualización de imágenes

Todas las entradas se convierten internamente a un array 2-D `float64` sin
normalizar su escala. Las imágenes `L` conservan sus intensidades; en `LA` se
usa sólo la luminancia; `RGB` se convierte mediante pesos de luminancia; y en
`RGBA` se ignora alpha antes de aplicar la misma conversión. Los TIFF grayscale
de 16 bits conservan sus valores. Las rutas y objetos PIL respetan además la
orientación EXIF.

Las imágenes de intensidad, la energía y los coefficient maps se renderizan en
grayscale por defecto. Esto afecta únicamente los archivos de visualización y
no modifica ningún array utilizado por la transformada. `theta` mantiene el
colormap cíclico `twilight`, porque representa ángulos y no intensidades.

`output_paths` admite, entre otras, las claves `original_image`,
`cartesian_coefficients`, `rotated_coefficients`, `theta`, `reconstruction`,
`reconstructed_image`, `coefficient_energy` y `metrics_csv`.

