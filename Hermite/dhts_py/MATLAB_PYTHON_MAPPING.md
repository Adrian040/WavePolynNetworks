# Correspondencia MATLAB → Python

El inventario se construyó a partir de todos los `.m` contenidos en
`dhts.zip`. Salvo las tres filas marcadas como prueba, cada función pública
está implementada en el archivo Python homólogo, no en un agregador auxiliar.

| MATLAB | Python | Estado |
|---|---|---|
| `angshow.m` | `angshow.py` | directa |
| `bincoef.m` | `bincoef.py` | directa |
| `binpyr.m` | `binpyr.py` | directa |
| `binpyr2.m` | `binpyr2.py` | directa |
| `bsmooth.m` | `bsmooth.py` | directa |
| `bsmooth2.m` | `bsmooth2.py` | directa; nombre de función MATLAB erróneo |
| `bt2dht.m` | `bt2dht.py` | directa |
| `chtmtx.m` | `chtmtx.py` | directa |
| `clssplot.m` | `clssplot.py` | directa |
| `Contents.m` | `Contents.py` | directa/documental |
| `ddht.m` | `ddht.py` | directa |
| `dht.m` | `dht.py` | directa; diferencia `dim` documentada |
| `dht2.m` | `dht2.py` | directa |
| `dht2_test.m` | `tests/test_dht2_reconstruction.py` | prueba |
| `dht2bt.m` | `dht2bt.py` | directa |
| `dht3.m` | `dht3.py` | directa; caso `D>N` pendiente |
| `dhtentr.m` | `dhtentr.py` | directa |
| `dhtgi.m` | `dhtgi.py` | directa |
| `dhti.m` | `dhti.py` | directa; errata `sumpot` documentada |
| `dhti2.m` | `dhti2.py` | directa |
| `dhtJ.m` | `dhtJ.py` | alias de la traducción `dht`, como el duplicado MATLAB |
| `dhtmorph.m` | `dhtmorph.py` | directa |
| `dhtmtx.m` | `dhtmtx.py` | directa |
| `dhtord.m` | `dhtord.py` | directa; caso 3‑D `D>N` pendiente |
| `dhtqt.m` | `dhtqt.py` | directa |
| `dhtshow.m` | `dhtshow.py` | directa |
| `dob2.m` | `dob2.py` | directa |
| `edht.m` | `edht.py` | directa |
| `energy.m` | `energy.py` | directa |
| `equaliz.m` | `equaliz.py` | directa |
| `fbt.m` | `fbt.py` | directa |
| `fbt2.m` | `fbt2.py` | directa |
| `fbt3.m` | `fbt3.py` | directa |
| `fbtmtx.m` | `fbtmtx.py` | directa |
| `gauge.m` | `gauge.py` | directa; rama `L>N` pendiente |
| `gbtmtx.m` | `gbtmtx.py` | directa; nombre de función MATLAB erróneo |
| `grafica.m` | `grafica.py` | directa |
| `graficaMapCoefs.m` | `graficaMapCoefs.py` | directa |
| `guidht.m` | `guidht.py` | núcleo visual equivalente; GUIDE no portable |
| `hermite.m` | `hermite.py` | directa |
| `hermiteFiltersFreq.m` | `hermiteFiltersFreq.py` | directa |
| `idht.m` | `idht.py` | directa |
| `idht2.m` | `idht2.py` | directa |
| `idht3.m` | `idht3.py` | directa; caso 3‑D `D>N` pendiente |
| `idhtqt.m` | `idhtqt.py` | directa |
| `im2qtb.m` | `im2qtb.py` | directa |
| `imcorn.m` | `imcorn.py` | directa; errata multibanda documentada |
| `imdht.m` | `imdht.py` | directa |
| `imdht2.m` | `imdht2.py` | directa |
| `lorient.m` | `lorient.py` | directa; argumento omitido en MATLAB documentado |
| `matshow.m` | `matshow.py` | directa |
| `mddht2.m` | `mddht2.py` | directa; nombre de función MATLAB erróneo |
| `mdht.m` | `mdht.py` | directa |
| `mdht2.m` | `mdht2.py` | directa |
| `mdhti.m` | `mdhti.py` | directa |
| `mdhti2.m` | `mdhti2.py` | directa |
| `mgauge.m` | `mgauge.py` | directa; variable MATLAB errónea documentada |
| `mrdht.m` | `mrdht.py` | directa |
| `mrdht2.m` | `mrdht2.py` | directa |
| `mscode.m` | `mscode.py` | directa |
| `obtainOrdCoefs.m` | `obtainOrdCoefs.py` | directa |
| `overshoot.m` | `overshoot.py` | directa |
| `pdht.m` | `pdht.py` | directa |
| `pred_test.m` | `tests/test_prediction.py` | prueba |
| `qdht.m` | `qdht.py` | directa |
| `qtb2im.m` | `qtb2im.py` | directa |
| `qtplot.m` | `qtplot.py` | directa |
| `rdht.m` | `rdht.py` | directa |
| `rdht2.m` | `rdht2.py` | directa |
| `rot_test.m` | `tests/test_rotation.py` | prueba |
| `samplat.m` | `samplat.py` | directa |
| `sdht2.m` | `sdht2.py` | directa salvo datos externos ausentes |
| `xdht2.m` | `xdht2.py` | directa |
| `zcross.m` | `zcross.py` | directa salvo `scest.mat` ausente |

## Archivos Python sin homólogo MATLAB

| Python | Motivo |
|---|---|
| `__init__.py` | API pública del paquete. |
| `__main__.py` | entrada `python -m dhts_py`. |
| `_core.py` | helpers privados de interoperabilidad NumPy/SciPy compartidos. |
| `_quadtree.py` | helpers privados de transformada/recorrido de bloques compartidos. |
| `tests/test_core.py` | regresión Python. |
| `tests/test_properties.py` | propiedades matemáticas. |
| `tests/test_matlab_equivalence.py` | comparación independiente MATLAB ↔ Python. |
| `tests/test_dht2_lena.py` | ejemplo visual DHT conservado. |
| `tests/test_idht_lena.py` | ejemplo visual IDHT conservado. |
| `tests/matlab/generate_matlab_references.m` | exportador de referencias para la suite Python. |
