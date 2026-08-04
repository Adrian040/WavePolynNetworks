# Notas de migración y trazabilidad

## Criterio numérico

El núcleo no usa aproximaciones Hermite continuas: construye los mismos filtros binomiales/Krawtchouk por convoluciones sucesivas y aplica la normalización de `dhtmtx.m`. La síntesis conserva la interpolación especial para `T<=2` y el cálculo general de pesos para `T>2`. `rdht` implementa la recurrencia normalizada del original, no una matriz de rotación genérica.

## Erratas inequívocas reparadas

- `gbtmtx.m` declaraba por error una función llamada `dhtmtx`; el módulo Python exporta `gbtmtx`.
- `bsmooth2.m` declaraba `bsmooth`; el módulo Python exporta `bsmooth2`.
- `mddht2.m` declaraba `mrdht2`; el módulo Python exporta `mddht2` y aplica `ddht`, que es la intención del archivo.
- `guidht.m` declaraba `guidhtq`; se exportan ambos nombres, `guidht` y `guidhtq`.
- `dht.m` contenía modificaciones `JOM` que accedían a `dim` antes de asignar su valor por defecto y luego lo reemplazaban por todos los ejes no unitarios. Se restauró el comportamiento descrito en la cabecera: primer eje no unitario o el `dim` indicado.
- `dhti.m` escribía `sumpot` pero usaba `sumopt`; se corrigió a `sumopt`.
- `mgauge.m` referenciaba `Y` aunque el argumento se llamaba `y`; el port usa el argumento recibido.
- `lorient.m` llamaba `mdhti2` sin `xsiz`; el port pasa el tamaño de la imagen.
- `imcorn.m` dependía de la interpretación multibanda de un arreglo de productos; el port suaviza cada producto explícitamente y conserva la fórmula de respuesta.

## Decisiones de interfaz

Python no tiene `nargout`. Las funciones con salidas opcionales usan banderas explícitas (`return_lowpass`, `return_theta`, `return_aux`, `return_bounds`). Los argumentos posicionales al estilo MATLAB siguen aceptándose en los casos principales, incluidos `shape` seguido de código de postproceso.

La GUI GUIDE de 1500 líneas no es portable fuera de MATLAB. `guidht.py` proporciona un explorador Matplotlib funcional con imagen, coeficientes, reconstrucción y controles `N/D/T`; el núcleo numérico que utiliza sí es el mismo del paquete.

## Datos externos no entregados

`pdcentr.mat` y `scest.mat` no estaban en el directorio suministrado. No se inventaron coeficientes entrenados. La clasificación perceptual tiene un reemplazo determinista; la estimación opcional de `zcross` exige que el usuario suministre el modelo. También faltaban `nei2band`, `vquant`/`vquantiz` y `normalize01`; sus operaciones necesarias se implementaron localmente.

## Precisión y bordes

Para `shape="full"` y base completa, las pruebas exigen reconstrucción con error máximo menor que `1e-12`. Los modos recortados (`same`, `valid`) y los modos que sólo prolongan el coeficiente de orden cero en la síntesis reproducen la pérdida de información de borde del algoritmo original y no prometen reconstrucción perfecta en el contorno.

