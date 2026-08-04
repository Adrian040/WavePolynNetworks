# DHTS Python

Port a Python/NumPy del *Discrete Hermite Transform Toolbox* (última versión MATLAB observada: diciembre de 2005). La prioridad del port es mantener las fórmulas, la normalización, el orden de los coeficientes, el submuestreo y las convenciones de borde del código suministrado.

## Inicio rápido

Desde la carpeta que contiene `dhts_py`:

```python
import numpy as np
from PIL import Image
from dhts_py import dht2, idht2, rdht, dhtshow

X = np.asarray(Image.open("dhts_py/lena.jpg").convert("L"), dtype=float) / 255
N, D, T = 8, 16, 2

Y = dht2(X, N, D, T)              # coeficientes cartesianos
Yr, theta = rdht(Y, N, D, "fwd", "grad", return_theta=True)
X2 = idht2(Y, X.shape, N, D, T)   # reconstrucción

print(np.max(np.abs(X - X2)))
```

Con la base completa (`D=2*N` en 2-D, `D=3*N` en 3-D), `shape="full"` y los mismos `N`, `D`, `T`, la ida y vuelta directa/inversa alcanza precisión de punto flotante.

## Convenciones

- Los parámetros conservan los nombres MATLAB: `N` es el orden/escala, `D` el orden total máximo y `T` la distancia de muestreo.
- `dim` conserva la numeración MATLAB empezando en 1. Los ejes ordinarios de NumPy no se usan en esta firma para evitar ambigüedad al migrar código.
- Los canales de coeficientes siempre ocupan el último eje NumPy.
- `dhtord(N, D, K)` es la fuente de verdad para asociar canal y orden.
- Los arreglos se calculan en `float64`; no hay *clipping* implícito.
- Los modos `full`, `same`, `valid`, `symm`, `repeat`, `asymm` y `cyclic` se escriben igual que en el toolbox cuando la función original los admite.

## Correspondencia de llamadas

```python
# MATLAB: H = dhtmtx(N,D)
H = dhtmtx(N, D)

# MATLAB: [H,G] = dhtmtx(N,D,T)
H, G = dhtmtx(N, D, T)

# MATLAB: [YH,Y0] = dht(...)
YH, Y0 = dht(..., return_lowpass=True)

# MATLAB: [Yr,theta] = rdht(...)
Yr, theta = rdht(..., return_theta=True)

# MATLAB: [Y,aux] = dht2(...)
Y, aux = dht2(..., return_aux=True)
```

Cada archivo `.m` suministrado tiene un módulo `.py` homólogo. Las implementaciones compartidas viven en `_core.py`, `_steering.py`, `_multiscale.py`, `_quadtree.py`, `_misc.py` y `_viz.py`; los módulos homólogos exportan la función con el nombre esperado.

## Pruebas

```powershell
python -m unittest discover -s dhts_py/tests -v
python -m dhts_py
```

Las pruebas verifican filtros binomiales, paridad, reconstrucción 1-D/2-D/3-D, involución FBT, ida y vuelta RDHT, quadtree y presencia de los 74 módulos homólogos.

## Dependencias externas ausentes en el material

Dos ramas del MATLAB original cargan archivos que no estaban entre los archivos entregados:

- `sdht2.m` carga `pdcentr.mat` para una clasificación vectorial entrenada. El port usa una clasificación determinista documentada basada en la proporción de energía 1-D/total cuando se pide `sdht2(Y,N,D,None,None)`.
- `zcross.m` carga `scest.mat` para estimar escala y contraste. La máscara de cruces por cero funciona sin él; para las salidas de escala/contraste se debe pasar el arreglo del modelo con `scale_model=`.

Consulta `PORTING_NOTES.md` para las erratas reparadas y los límites de equivalencia.

