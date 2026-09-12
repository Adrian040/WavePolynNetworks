# Transformada Hermite--Gaussiana 2-D

Esta carpeta contiene una implementación modular de una transformada local de
Hermite para imágenes. El objetivo es obtener, en cada posición de una
retícula espacial, coeficientes que describan la intensidad y sus variaciones
locales mediante polinomios de Hermite modulados por una ventana Gaussiana.

La implementación incluye:

- análisis cartesiano separable;
- estimación de la dirección local del gradiente;
- *steering* de los coeficientes hacia esa dirección;
- *steering* inverso;
- síntesis truncada por *overlap-add*;
- diagnósticos de los filtros, métricas y figuras reproducibles.

## Definición empleada

El filtro unidimensional de orden `n` es

\[
a_n(x;\sigma)=
\frac{H_n(x/\sigma)}{\sqrt{2^n n!}}
\frac{\exp[-(x/\sigma)^2]}{\sigma\sqrt{\pi}},
\]

donde \(H_n\) es el polinomio de Hermite de físicos y `sigma` se expresa en
píxeles. El filtro bidimensional es separable:

\[
D_{mn}(x,y;\sigma)=a_m(x;\sigma)a_n(y;\sigma).
\]

Los coeficientes \(L_{mn}\) se calculan correlacionando la imagen con
\(D_{mn}\). El primer índice `m` siempre corresponde a columnas/eje `x`; el
segundo índice `n`, a filas/eje `y`. Las intensidades de entrada no se
normalizan: una imagen en 0--255 y otra en 0--1 producen coeficientes con
escalas diferentes.

Las funciones continuas anteriores se muestrean sobre coordenadas enteras y
se truncan a un soporte impar. El radio crece automáticamente hasta que varias
muestras consecutivas de las colas de todos los órdenes sean menores que una
tolerancia relativa de `1e-8`.

## Diferencia respecto a construcciones discretas binomiales

Esta implementación parte directamente de funciones Hermite--Gaussianas
continuas y después las muestrea en la retícula de píxeles. En otros trabajos
se sigue un acercamiento propiamente discreto: los filtros se construyen a
partir de suavizados y diferencias binomiales finitas, y su estructura está
relacionada con polinomios de Krawtchouk.

Las dos familias están relacionadas en un sentido límite, pero no son
intercambiables parámetro por parámetro:

- aquí `sigma` controla directamente la escala de la Gaussiana continua;
- en la construcción binomial la escala depende de la longitud y de los
  parámetros del banco discreto;
- el banco binomial puede diseñarse con ortogonalidad y filtros de síntesis
  exactos dentro de un espacio discreto finito;
- el banco de esta carpeta es una aproximación muestreada y truncada, por lo
  que su calidad discreta debe vigilarse con los diagnósticos incluidos;
- los valores numéricos de sus coeficientes no deben compararse directamente
  sin reconciliar normalización, escala y convención de síntesis.

La elección depende del objetivo. La formulación actual conserva de manera
explícita la interpretación de escala espacial de las funciones continuas y
facilita el *steering* por orden total. Una formulación binomial es conveniente
cuando la prioridad es una transformada finita puramente discreta con un par
análisis--síntesis diseñado conjuntamente.

## Regiones de coeficientes

`coefficient_region="triangle"` conserva

\[
m+n\leq \texttt{max_order}.
\]

`coefficient_region="square"` conserva

\[
0\leq m,n\leq \texttt{max_order}.
\]

El *steering* solo es correcto sobre bloques completos de igual orden total.
Por ello, el modo cuadrado calcula internamente el triángulo completo hasta
grado `2*max_order`; después devuelve únicamente los pares del cuadrado. Estos
canales auxiliares son deliberados y no deben eliminarse para ahorrar cálculo.

## Orientación y steering

La dirección dominante se estima mediante

\[
\theta=\operatorname{atan2}(L_{01},L_{10}).
\]

`theta` está en radianes y representa la dirección del gradiente, es decir, la
normal a un borde. La dirección tangente al borde difiere en \(\pi/2\). El
operador de *steering* mezcla únicamente coeficientes con el mismo grado total
y respeta la normalización Hermite. Aplicar el mismo operador con `-theta`
recupera los coeficientes cartesianos hasta error de punto flotante.

También puede utilizarse `rotation_mode="fixed"` con un ángulo escalar o un
mapa angular, en grados o radianes.

## Síntesis implementada

La reconstrucción coloca los coeficientes sobre la retícula de paso `T =
sampling_step`, convoluciona cada canal con los mismos filtros del análisis y
suma sus contribuciones. Finalmente divide entre la suma de las ventanas de
orden cero desplazadas:

\[
\widehat I(x,y)=
\frac{
\sum_{p,q}\sum_{(m,n)\in\mathcal R}
L_{mn}[p,q]a_m(x-qT)a_n(y-pT)
}{
\sum_{p,q}a_0(x-qT)a_0(y-pT)
}.
\]

En esta convención, el producto bidimensional de orden cero corresponde a la
ventana denominada `window_squared` o \(w^2\). Como \(\mathcal R\) contiene un
número finito de órdenes, la salida es una **síntesis truncada**, no una
afirmación de inversión exacta de la expansión infinita.

## Bordes y geometría espacial

El parámetro `boundary` controla la extensión durante el análisis:

| Valor | Comportamiento |
|---|---|
| `symmetric` | Reflexión simétrica; opción predeterminada. |
| `edge` | Repite el valor de la fila o columna más cercana. |
| `constant` | Usa cero fuera de la imagen. |
| `wrap` | Extensión circular. |

El análisis conserva geometría `same` antes del submuestreo. `same` no es una
extensión adicional. El modo `valid` no se expone porque cambiaría el origen y
el tamaño de la retícula, además de requerir una síntesis distinta.

## Discretización y elección de sigma

Una cola pequeña no garantiza por sí sola ortogonalidad después del muestreo.
`build_filter_bank` reporta para cada orden:

- suma firmada y fuga DC relativa;
- error máximo de paridad;
- magnitud relativa en el extremo del soporte.

Los órdenes pares superiores pueden responder apreciablemente a una imagen
constante cuando `sigma` es demasiado pequeña respecto al píxel. El código
emite una advertencia si la fuga DC relativa supera `1e-3`. El valor
predeterminado `sigma=2.0` tuvo fuga despreciable en las pruebas de esta
implementación. Para otras escalas y órdenes se deben conservar y revisar los
diagnósticos en `result["filter_bank"]["diagnostics"]`; no se renormalizan los
filtros silenciosamente porque eso cambiaría la transformada y su *steering*.

## Estructura

```text
Refactored_HT_final/
├── hermite_transform/
│   ├── __init__.py       # API pública
│   ├── image_io.py       # lectura y grayscale
│   ├── filters.py        # órdenes, soporte y banco
│   ├── analysis.py       # transformada cartesiana
│   ├── steering.py       # orientación y rotación
│   ├── synthesis.py      # reconstrucción truncada
│   ├── metrics.py        # métricas, stack y energía
│   ├── visualization.py  # guardado de figuras y CSV
│   └── workflow.py       # flujo completo
├── tests/
├── example.py
├── requirements.txt
└── README.md
```

No existe un módulo separado de validaciones. Cada etapa conserva solamente
las comprobaciones que evitan resultados ambiguos o silenciosamente inválidos.

## Instalación

Desde esta carpeta:

```bash
python -m pip install -r requirements.txt
```

La versión se verificó con Python 3.11, NumPy 1.26.4, SciPy 1.11.4, Pillow
10.2.0 y Matplotlib 3.8.0.

## Uso rápido

Ejecute:

```bash
python example.py
```

El ejemplo usa `../Fusion_Images_ds/house.tif`. Debajo queda preparada y
comentada la ruta para `lena.jpg`.

Uso desde otro script:

```python
from hermite_transform import hermite_transform_image

result = hermite_transform_image(
    image="../Fusion_Images_ds/house.tif",
    max_order=3,
    sigma=2.0,
    coefficient_region="square",
    sampling_step=1,
    boundary="symmetric",
    rotation_mode="dominant",
    results_path="results_house",
)
```

Para trabajar etapa por etapa:

```python
from hermite_transform import (
    build_filter_bank,
    cartesian_transform,
    dominant_theta,
    inverse_rotate_coefficients,
    read_image,
    rotate_coefficients,
    synthesize,
)

image = read_image("../Fusion_Images_ds/house.tif")
bank = build_filter_bank(max_order=3, sigma=2.0, region="square")
complete = cartesian_transform(image, bank, orders=bank["steering_orders"])
theta = dominant_theta(complete)
rotated = rotate_coefficients(complete, theta, bank["steering_orders"])
recovered = inverse_rotate_coefficients(
    rotated, theta, bank["steering_orders"]
)
public = {order: recovered[order] for order in bank["orders"]}
reconstruction = synthesize(public, bank, image.shape)
```

## Archivos de resultados

`hermite_transform_image` crea `results_path` y, con el flujo predeterminado,
genera:

1. `01_original_input_grayscale_image.png`
2. `02_cartesian_hermite_coefficient_maps.png`
3. `03_steered_rotated_hermite_coefficient_maps.png`
4. `04_local_gradient_orientation_theta_in_degrees.png`
5. `05_recovered_cartesian_from_rotated_coefficients.png`
6. `06_truncated_hermite_synthesis_reconstructed_image.png`
7. `07_reconstruction_comparison_with_absolute_error.png`
8. `08_rotated_hermite_coefficient_energy_without_dc.png`
9. `09_steering_roundtrip_and_reconstruction_metrics.csv`

Si se desactiva una etapa, no se genera el archivo que depende de ella. El
mapping `result["saved_files"]` indica exactamente qué se escribió.

## Resultados disponibles en memoria

El resultado principal incluye:

- `cartesian_coefficients`, `rotated_coefficients` y
  `recovered_cartesian_coefficients`;
- `theta` en radianes;
- `reconstructed_image` y sus métricas;
- `coeff_stack`, con canales en el orden de `result["orders"]`;
- `coefficient_energy`, excluyendo L00;
- órdenes visibles y auxiliares;
- banco, soporte y diagnósticos;
- rutas de salida.

## Pruebas

Desde `Refactored_HT_final`:

```bash
python -m unittest discover -s tests -v
```

Las pruebas cubren profundidad de bits, paridad, fuga DC, convención de ejes,
modos de borde, *steering* de primer orden y de orden alto, síntesis y los nueve
archivos del flujo completo.

## Reproducibilidad y uso académico

Al reportar resultados conviene registrar al menos:

- versión de Python y dependencias;
- escala original de intensidad de las imágenes;
- `max_order`, `sigma`, región, `sampling_step` y borde;
- modo y unidad de la orientación;
- radio de soporte y diagnósticos de filtros;
- lista y orden de canales utilizados;
- métricas de steering y reconstrucción.

Antes de reutilizar el código en una publicación también debe incorporarse la
referencia bibliográfica concreta que motive la normalización y la convención
de síntesis empleadas, además de la licencia de distribución que corresponda.
