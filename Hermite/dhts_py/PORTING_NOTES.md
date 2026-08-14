# Notas de port y diferencias verificables

Este archivo registra únicamente diferencias reales entre el MATLAB
suministrado y Python, erratas del material y dependencias ausentes. Los tests
internos no se presentan como evidencia de equivalencia MATLAB.

## Erratas de nombres o ejecución ya presentes en MATLAB

- `gbtmtx.m` declara una función llamada `dhtmtx`; Python exporta `gbtmtx`.
- `bsmooth2.m` declara `bsmooth`; Python exporta `bsmooth2`.
- `mddht2.m` declara `mrdht2`; Python exporta `mddht2` y aplica `ddht`, que es
  la intención indicada por el nombre del archivo.
- `guidht.m` declara `guidhtq`; Python conserva los alias `guidht` y
  `guidhtq`.
- `dhti.m` escribe `sumpot` pero usa `sumopt`.
- `mgauge.m` referencia `Y` aunque el argumento se llama `y`.
- `lorient.m` llama `mdhti2` sin el argumento `xsiz`.

## Divergencias matemáticas preexistentes pendientes de decisión

No se modificaron durante esta refactorización.

### `gauge` cuando `L > N`

En la rama inferior a la antidiagonal, `gauge.m` construye un bloque de grado
`nn=N-n`, pero divide `atan2(a,b)` por el contador `n`. `gauge.py` divide por
el grado efectivo del bloque (`nn`). Para las validaciones pedidas (`N=8`,
`D=3`, `L=1`) esta rama no se usa; para `D>N` puede producir ángulos distintos.
Se requiere decisión explícita antes de cambiar uno de los comportamientos.

### `dhtord`/`dht3` con `D > N`

`dhtord.m` permite que la tercera coordenada llegue hasta `D`, mientras
`dhtmtx.m` sólo produce filtros hasta `N`; por ello `dht3.m` puede intentar
indexar una columna inexistente cuando `D>N`. Python limita cada orden por eje
y permite la base 3‑D completa. El roundtrip 3‑D actual es una prueba interna
Python, no una equivalencia validada contra MATLAB.

### Selección de `dim` en `dht.m`

El archivo suministrado contiene líneas marcadas `JOM` que leen `dim` antes de
establecer el valor por defecto y después lo reemplazan por todos los ejes no
unitarios. Python conserva la interfaz descrita en la cabecera: usa el `dim`
solicitado o el primer eje no unitario. Esta diferencia de control de ejes no
se cambió en esta tarea.

## Datos externos no suministrados

- `sdht2.m` requiere `pdcentr.mat` y funciones de cuantización ausentes. La
  rama adaptativa Python contiene un fallback determinista preexistente; no es
  evidencia de equivalencia y debe considerarse pendiente hasta disponer de
  esos datos.
- `zcross.m` requiere `scest.mat` para escala/contraste. Python exige pasar el
  modelo mediante `scale_model=`; no inventa sus coeficientes.
- También faltan `nei2band`, `vquant`/`vquantiz` y `normalize01`; sólo las
  operaciones necesarias para las rutas disponibles se implementaron
  localmente.

## Diferencias inevitables de interfaz

Python no tiene `nargout`. Las salidas opcionales usan `return_lowpass`,
`return_theta`, `return_aux` y `return_bounds`. Los canales ocupan el último
eje NumPy y todos los cálculos científicos se conservan en `float64`.

La GUI MATLAB GUIDE de aproximadamente 1500 líneas no es portable sin GUIDE.
`guidht.py` ofrece un explorador Matplotlib de carga, transformación,
coeficientes y reconstrucción; no pretende reproducir la interfaz gráfica.

## Estado de equivalencia

En este entorno no se encontró MATLAB ni GNU Octave y no se entregaron arrays
MATLAB (`.mat`/CSV) para las diez comparaciones. La suite de equivalencia queda
preparada y se marca `PENDING MATLAB REFERENCE` hasta ejecutar
`tests/matlab/generate_matlab_references.m`. Los PNG y la tabla de métricas ya
existentes no se usan como referencia numérica de coeficientes.
