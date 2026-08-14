# DHTS Python

Port Python/NumPy del *Discrete Hermite Transform Toolbox* de MATLAB. La
prioridad es reproducir de forma trazable los filtros binomiales/Krawtchouk,
normalización, orden de coeficientes, muestreo, bordes, steering y síntesis del
toolbox original suministrado; no sustituirlos por formulaciones Hermite
continuas ni por aproximaciones de otras bibliotecas.

## Estructura y trazabilidad

Cada función pública vive en el archivo homólogo al `.m`:

```text
dhts_py/
├── dhtmtx.py          # dhtmtx.m
├── dhtord.py          # dhtord.m
├── dht.py             # dht.m
├── dht2.py            # dht2.m
├── gauge.py           # gauge.m
├── rdht.py            # rdht.m
├── idht.py            # idht.m
├── idht2.py           # idht2.m
├── ...                # resto de homólogos MATLAB
├── _core.py           # sólo helpers privados de arrays/bordes/síntesis
├── _quadtree.py       # sólo helpers privados compartidos de quadtree
├── tests/
│   ├── test_core.py
│   ├── test_properties.py
│   ├── test_matlab_equivalence.py
│   ├── matlab/
│   │   └── generate_matlab_references.m
│   └── resultados/
│       ├── dht/
│       └── idht/
└── README.md
```

Los antiguos agregadores `_misc.py`, `_multiscale.py`, `_steering.py` y
`_viz.py` se eliminaron. `_core.py` se conserva para operaciones sin homólogo
MATLAB que comparten varias traducciones (validación escalar, padding,
convolución por eje y geometría de upsampling). `_quadtree.py` conserva la
matriz de bloques, el lector de bits y la rotación de bloque que comparten
`dhtqt.py`, `idhtqt.py`, `im2qtb.py`, `qtb2im.py` y `qtplot.py`; los algoritmos
públicos permanecen en esos cinco archivos.

Los scripts MATLAB `dht2_test.m`, `pred_test.m` y `rot_test.m` son pruebas, no
funciones públicas. Sus equivalentes Python están en `tests/`.

## Convenciones

- `N` es el parámetro de escala; los filtros tienen longitud `N+1`.
- `D` es el orden total máximo. En 2‑D, `D=2*N` incluye el cuadro completo
  `0 <= m,n <= N`.
- `T` es la distancia de muestreo.
- Los coeficientes ocupan el último eje NumPy y usan `float64`.
- `dim` conserva la indexación MATLAB empezando en 1.
- `dhtord.m`/`dhtord.py` son la fuente de verdad del orden de canales:
  `L00, L10, L01, L20, L11, L02, ...`.
- Los modos disponibles son `full`, `same`, `valid`, `symm`, `repeat`,
  `asymm` y, donde el port lo admite, `cyclic`.
- No hay clipping ni reescalado en los datos científicos. `dhtshow` y otras
  rutinas normalizan copias destinadas sólo a visualización.

## Instalación

Desde la carpeta que contiene `dhts_py`:

```powershell
python -m pip install -r dhts_py/requirements.txt
```

Las dependencias son NumPy, SciPy, Matplotlib y Pillow.

## Uso mínimo

```python
import numpy as np
from dhts_py import dht2, idht2

X = np.random.default_rng(1).normal(size=(32, 32)).astype(np.float64)
N, D, T = 8, 16, 1
Y = dht2(X, N, D, T, "full")
X_reconstructed = idht2(Y, X.shape, N, D, T, "full")
```

## Tres categorías de pruebas

Ejecutar todo:

```powershell
python -m unittest discover -s dhts_py/tests -v
```

Las categorías se mantienen separadas:

1. `test_core.py`: regresión del port Python.
2. `test_properties.py`: propiedades matemáticas (DC, orden, rampas,
   steering y reconstrucción completa frente a truncada).
3. `test_matlab_equivalence.py`: comparación exclusiva con arrays producidos
   independientemente por MATLAB. Si el `.mat` no existe, cada prueba aparece
   como `PENDING MATLAB REFERENCE`/`skipped`; nunca se usa Python como supuesta
   referencia MATLAB.

## Pruebas visuales

```powershell
python -m dhts_py.tests.test_dht2_lena --no-show
python -m dhts_py.tests.test_idht_lena --no-show
```

La primera conserva las vistas cartesiana, rotada por gradiente y
multidireccional por Hessiano. La segunda guarda original, coeficientes y
reconstrucciones. Sin `--save-dir`, las salidas van a:

```text
dhts_py/tests/resultados/dht/
dhts_py/tests/resultados/idht/
```

`--save-dir` permite cambiar la subcarpeta de salida explícitamente. Para no
ensuciar el repositorio, use siempre una ruta dentro de `tests/resultados/`.

## Validación MATLAB ↔ Python

El script `tests/matlab/generate_matlab_references.m` genera un MAT v7 con las
diez referencias requeridas:

1. `dhtmtx` para `(N,D)=(3,3),(4,4),(8,3),(8,8)`, incluidos `G` con `T=1`.
2. `dhtord(8,3,2)`.
3. Imagen constante.
4. Rampa horizontal.
5. Rampa vertical.
6. `house.tif`, `N=8`, `D=3`, `T=1`, `symm`, canal por canal.
7. `gauge` de primer orden con error angular envuelto módulo `2*pi`.
8. RDHT directa con los mismos cartesianos y `theta`.
9. RDHT inversa y roundtrip Python por separado.
10. DHT+IDHT completa (`D=16`) y truncada (`D=3`).

Ejemplo en MATLAB, usando una copia extraída y sin modificar del toolbox:

```matlab
addpath('C:\ruta\al\repo\dhts_py\tests\matlab')
generate_matlab_references( ...
    'C:\ruta\al\toolbox\dhts', ...
    'C:\ruta\al\repo\Fusion_Images_ds\house.tif')
```

Esto crea por defecto:

```text
dhts_py/tests/matlab/references/dhts_matlab_reference.mat
```

Después, ejecute la suite Python. También puede apuntar a otro archivo:

```powershell
$env:DHTS_MATLAB_REFERENCE='C:\ruta\referencia.mat'
python -m unittest dhts_py.tests.test_matlab_equivalence -v
```

Las comparaciones usan `atol=rtol=5e-13` en `float64` y reportan
`max_abs_error`, `mean_abs_error`, `MSE`, `RMSE` y `np.allclose`. Cuando hay
referencia, las métricas se guardan en
`tests/resultados/matlab/metrics.json`.

## Reconstrucción completa y truncada

Para `N=8` en 2‑D:

- `D=3` conserva diez canales hasta orden total tres. Es una expansión
  truncada y no debe exigirse reconstrucción perfecta.
- `D=16` incluye los 81 pares `0 <= m,n <= 8`, incluido `L88`. Con `full` y
  parámetros compatibles, el roundtrip Python alcanza precisión de punto
  flotante; la equivalencia MATLAB sólo se declara cuando el MAT independiente
  también coincide.

Las divergencias reales o pendientes están documentadas en
`PORTING_NOTES.md`. La tabla exhaustiva de archivos está en
`MATLAB_PYTHON_MAPPING.md`.
