# Transformada Discreta de Hermite

`dhts_py` implementa una representación local de imágenes mediante un filter
bank discreto de Hermite. La transformada cartesiana produce coefficient maps
`L_{m,n}`; a partir de ellos se puede estimar una orientación local, aplicar
steering mediante la RDHT, recuperar los coeficientes cartesianos y sintetizar
una imagen con la IDHT.

El banco 1-D parte de una ventana binomial de orden cero. Los filtros de orden
superior forman la representación discreta relacionada con polinomios de
Krawtchouk y aproximan el comportamiento de las funciones Hermite--Gaussianas.
Los filtros 2-D se obtienen de manera separable.

## API

La API pública del paquete contiene solamente:

```python
from dhts_py import dhtmtx, dhtord, dht2, gauge, rdht, idht2
```

Las funciones de visualización y métricas se importan desde `dhts_py.utils`.
Los cálculos principales usan `float64`, no normalizan la imagen y esperan un
array grayscale 2-D.

## Parámetros

- `N >= 0` fija la escala y el máximo orden individual disponible. Los filtros
  1-D tienen longitud `N + 1` y `0 <= m,n <= N`.
- `D >= 0` determina qué órdenes se conservan. Su significado depende de
  `coefficient_region`.
- `T >= 1` es el paso espacial de sampling. `T=1` conserva una muestra por
  posición; un valor mayor realiza subsampling.
- `coefficient_region` puede ser `"triangle"` o `"square"`.

La transformada usa convolución `full`. Para una imagen de shape `(H, W)`, el
stack tiene shape `(ceil((H+N)/T), ceil((W+N)/T), C)`. El índice `m` corresponde
al eje horizontal y `n` al vertical.

### Triangle

```python
N = 8
D = 3
coefficient_region = "triangle"
```

Selecciona `0 <= m,n <= N` y `m+n <= D`. `D` no puede exceder `2*N`. Los
canales se agrupan por orden total:

```text
(0,0)
(1,0), (0,1)
(2,0), (1,1), (0,2)
(3,0), (2,1), (1,2), (0,3)
```

En este ejemplo se obtienen 10 mapas. Una reconstrucción triangular truncada
es una aproximación; no se espera que coincida con la imagen original.

### Square parcial

```python
N = 8
D = 3
coefficient_region = "square"
```

Define `M=min(D,N)` y selecciona todos los pares `0 <= m,n <= M`. Por tanto,
se obtienen 16 mapas, desde `L_{0,0}` hasta `L_{3,3}`. Internamente siguen
agrupados por orden total; `dhtord` es siempre la fuente de verdad del canal.

### Square completo

```python
N = 8
D = 8
coefficient_region = "square"
```

Incluye los 81 pares `0 <= m,n <= 8`. Éste es el modo recomendado para
comprobar la reconstrucción completa con el banco disponible.

## DHT triangular y visualización

```python
import numpy as np

from dhts_py import dht2, dhtord
from dhts_py.utils import plot_coefficients

image = np.random.default_rng(1).normal(size=(32, 32))
orders = dhtord(N=8, D=3, coefficient_region="triangle")
coeffs = dht2(
    image,
    N=8,
    D=3,
    T=1,
    coefficient_region="triangle",
)

figure, axes = plot_coefficients(
    coeffs,
    orders,
    coefficient_region="triangle",
)
```

`plot_coefficients` puede usar normalización `"individual"`, `"global"` o
`None`. La escala se aplica sólo a la visualización y nunca modifica `coeffs`.

## DHT square parcial

```python
from dhts_py import dht2, dhtord
from dhts_py.utils import plot_coefficients

orders = dhtord(N=8, D=3, coefficient_region="square")
coeffs = dht2(
    image,
    N=8,
    D=3,
    T=1,
    coefficient_region="square",
)

figure, axes = plot_coefficients(
    coeffs,
    orders,
    coefficient_region="square",
)
```

La cuadrícula usa columnas para `m` y filas para `n`; contiene 16 coefficient
maps.

## Reconstrucción completa

```python
from dhts_py import dht2, idht2
from dhts_py.utils import reconstruction_metrics

coeffs = dht2(
    image,
    N=8,
    D=8,
    T=1,
    coefficient_region="square",
)
reconstructed = idht2(
    coeffs,
    image.shape,
    N=8,
    D=8,
    T=1,
    coefficient_region="square",
)
metrics = reconstruction_metrics(image, reconstructed)
```

La síntesis suma exactamente los 81 mapas indicados por `dhtord`; no convierte
el square en triangle ni descarta términos con `m+n>D`. Las pequeñas diferencias
respecto a la entrada se deben al redondeo de punto flotante.

## Orientación y RDHT

```python
from dhts_py import dht2, gauge, rdht

coeffs = dht2(image, N=8, D=3, T=1, coefficient_region="triangle")
theta = gauge(
    coeffs,
    N=8,
    D=3,
    mode="gradient",
    coefficient_region="triangle",
)
rotated = rdht(
    coeffs,
    theta,
    N=8,
    D=3,
    direction="forward",
    coefficient_region="triangle",
)
recovered = rdht(
    rotated,
    theta,
    N=8,
    D=3,
    direction="inverse",
    coefficient_region="triangle",
)
```

`gauge` admite `mode="gradient"` y `mode="hessian"`. La RDHT consulta
`dhtord` para formar cada bloque de orden total, tanto en triangle como en
square.

## Utilidades

`dhts_py.utils` expone:

- `coefficient_index`: localiza el canal de un componente `L_{m,n}`.
- `plot_coefficients`: visualiza layouts triangle o square.
- `reconstruction_metrics`: calcula MSE, RMSE, MAE, error máximo y PSNR.
- `plot_reconstruction_comparison`: muestra original, reconstrucción y error.
- `compare_coefficients`: calcula errores por coefficient map.

## Dependencias y tests

La implementación requiere NumPy, SciPy y Matplotlib:

```powershell
python -m pip install numpy scipy matplotlib
python -m unittest discover -s dhts_py/tests -v
```
