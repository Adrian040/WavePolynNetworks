"""Transformada Discreta de Hermite cartesiana, rotada e inversa.

Esta implementación en Python reproduce el núcleo matemático del toolbox de
MATLAB proporcionado por los tutores:

* ``dhtmtx.m``  -> filtros binomiales/Krawtchouk de análisis e interpolación.
* ``dht2.m``    -> transformada cartesiana bidimensional.
* ``rdht.m``    -> steering normalizado de los coeficientes.
* ``idht2.m``   -> rotación inversa y síntesis de la imagen.

La función principal :func:`hermite_transform_image` permite ejecutar, desde
una sola interfaz, cualquiera de los siguientes recorridos:

1. Imagen -> coeficientes cartesianos.
2. Imagen -> cartesianos -> rotados.
3. Imagen -> cartesianos -> rotados -> cartesianos recuperados.
4. Imagen -> cartesianos -> imagen reconstruida.
5. Imagen -> cartesianos -> rotados -> cartesianos recuperados -> imagen.

Convenciones
------------
* Un orden se representa como ``(m, n)``:
  ``m`` es el orden en x y ``n`` es el orden en y.
* ``N`` determina la longitud de los filtros: ``N + 1``.
* ``D`` es el orden total máximo: ``m + n <= D``.
* ``T`` es la distancia de muestreo de los mapas de coeficientes.
* Para obtener el cuadro completo ``0 <= m,n <= N`` se requiere ``D = 2*N``.
* La reconstrucción perfecta del toolbox se obtiene, salvo redondeo numérico,
  con el cuadro completo, la misma ``N``, ``D`` y ``T`` en ambas direcciones,
  y ``shape='full'``.

Dependencias
------------
numpy, scipy, Pillow y matplotlib.
"""

from __future__ import annotations

import csv
from math import comb, ceil, floor
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import convolve2d


Array = np.ndarray
Orden = Tuple[int, int]
Coeficientes = Dict[Orden, Array]
Ruta = Union[str, Path]
ImagenEntrada = Union[Ruta, Image.Image, Array]


# =============================================================================
# Lectura de imagen y utilidades de rutas
# =============================================================================


def leer_imagen_gris(imagen: ImagenEntrada) -> Array:
    """Lee una imagen como arreglo 2-D ``float64`` en el intervalo [0, 1].

    La función contempla explícitamente imágenes TIFF de dos canales
    ``gris + alfa``; en ese caso conserva únicamente el canal de intensidad.

    Parameters
    ----------
    imagen:
        Ruta, imagen de PIL o arreglo NumPy.

    Returns
    -------
    numpy.ndarray
        Imagen en escala de grises, con forma ``(alto, ancho)``.
    """
    if isinstance(imagen, (str, Path)):
        with Image.open(imagen) as img_pil:
            # En una imagen LA, convert('L') conserva el canal de luminancia y
            # descarta correctamente el alfa.
            arreglo = np.asarray(img_pil.convert("L"))
    elif isinstance(imagen, Image.Image):
        arreglo = np.asarray(imagen.convert("L"))
    else:
        arreglo = np.asarray(imagen)
        if arreglo.ndim == 3:
            if arreglo.shape[-1] == 2:
                # Gris + alfa.
                arreglo = arreglo[..., 0]
            elif arreglo.shape[-1] >= 3:
                # Conversión de luminancia para RGB/RGBA.
                rgb = arreglo[..., :3].astype(np.float64)
                arreglo = (
                    0.2126 * rgb[..., 0]
                    + 0.7152 * rgb[..., 1]
                    + 0.0722 * rgb[..., 2]
                )
            elif arreglo.shape[-1] == 1:
                arreglo = arreglo[..., 0]

    if arreglo.ndim != 2:
        raise ValueError(
            "La imagen debe quedar en dos dimensiones; "
            f"se obtuvo la forma {arreglo.shape}."
        )
    if arreglo.size == 0:
        raise ValueError("La imagen de entrada está vacía.")

    tipo_original = arreglo.dtype
    arreglo = arreglo.astype(np.float64, copy=False)

    if np.issubdtype(tipo_original, np.integer):
        maximo_tipo = np.iinfo(tipo_original).max
        arreglo = arreglo / float(maximo_tipo)
    else:
        maximo = float(np.nanmax(arreglo))
        if maximo > 1.5:
            # Convención habitual para arreglos flotantes que contienen 0..255.
            divisor = 255.0 if maximo <= 255.0 else maximo
            arreglo = arreglo / divisor

    if not np.isfinite(arreglo).all():
        raise ValueError("La imagen contiene valores NaN o infinitos.")

    return arreglo


def _preparar_ruta(ruta: Optional[Ruta]) -> Optional[Path]:
    """Crea el directorio padre de una ruta de salida, si hace falta."""
    if ruta is None:
        return None
    salida = Path(ruta)
    salida.parent.mkdir(parents=True, exist_ok=True)
    return salida


# =============================================================================
# Órdenes y filtros binomiales de la DHT
# =============================================================================


def validar_parametros(N: int, D: int, T: int, shape: str) -> None:
    """Valida los parámetros principales de la transformada."""
    if not isinstance(N, (int, np.integer)) or N < 0:
        raise ValueError("N debe ser un entero no negativo.")
    if not isinstance(D, (int, np.integer)) or D < 0 or D > 2 * N:
        raise ValueError("D debe ser entero y cumplir 0 <= D <= 2*N.")
    if N == 0:
        if T != 1:
            raise ValueError("Para N=0 debe utilizarse T=1.")
    elif not isinstance(T, (int, np.integer)) or not (1 <= T <= N):
        raise ValueError("T debe ser entero y cumplir 1 <= T <= N.")

    if shape.lower() not in {"full", "same", "valid", "symm"}:
        raise ValueError("shape debe ser 'full', 'same', 'valid' o 'symm'.")


def ordenes_dht2(N: int, D: int) -> list[Orden]:
    """Devuelve los órdenes ``(m,n)`` en el mismo orden que ``dhtord.m``.

    ``m`` es el orden en x y ``n`` el orden en y. Para cada orden total se
    recorre primero el coeficiente más horizontal:

    ``(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...``
    """
    if N < 0 or D < 0 or D > 2 * N:
        raise ValueError("Se requiere N >= 0 y 0 <= D <= 2*N.")

    ordenes: list[Orden] = []
    for total in range(min(D, 2 * N) + 1):
        for orden_y in range(max(0, total - N), min(N, total) + 1):
            orden_x = total - orden_y
            ordenes.append((orden_x, orden_y))
    return ordenes


def _matriz_convolucion(vector: Array, numero_columnas: int) -> Array:
    """Equivalente mínimo de ``convmtx(vector, numero_columnas)`` de MATLAB."""
    vector = np.asarray(vector, dtype=np.float64).ravel()
    matriz = np.zeros(
        (vector.size + numero_columnas - 1, numero_columnas),
        dtype=np.float64,
    )
    for columna in range(numero_columnas):
        matriz[columna : columna + vector.size, columna] = vector
    return matriz


def matriz_dht_binomial(
    N: int,
    D: int,
    T: Optional[int] = None,
) -> Union[Array, Tuple[Array, Array]]:
    """Construye los filtros 1-D de análisis y, opcionalmente, de síntesis.

    Esta función es una traducción directa de ``dhtmtx.m``.

    Parameters
    ----------
    N:
        Parámetro de escala. La longitud de cada filtro es ``N+1``.
    D:
        Orden máximo solicitado. En una dimensión se usa ``min(D,N)``.
    T:
        Si se proporciona, también se calculan las funciones de interpolación
        empleadas por la transformada inversa.

    Returns
    -------
    H o (H, G):
        ``H[:,k]`` es el filtro de análisis de orden ``k``.
        ``G[:,k]`` es el filtro de síntesis/interpolación de orden ``k``.

    Notes
    -----
    Se les llama filtros binomiales porque el filtro de orden cero se obtiene
    al convolucionar repetidamente ``[1,1]/2`` y sus coeficientes son
    ``binom(N,k)/2**N``. Los órdenes superiores se generan introduciendo la
    máscara de diferencia ``[1,-1]/2``. Esta familia discreta está relacionada
    con los polinomios de Krawtchouk y converge, en el límite, a las funciones
    Hermite-Gaussianas continuas.
    """
    if not isinstance(N, (int, np.integer)) or N < 0:
        raise ValueError("N debe ser un entero no negativo.")
    if not isinstance(D, (int, np.integer)) or D < 0:
        raise ValueError("D debe ser un entero no negativo.")

    D_1d = min(N, D)

    if N == 0:
        H = np.ones((1, 1), dtype=np.float64)
    else:
        if D_1d > 0:
            # Columnas: máscara de suavizado y máscara de diferencia.
            B = np.array([[0.5, 0.5], [0.5, -0.5]], dtype=np.float64)
        else:
            B = np.array([[0.5], [0.5]], dtype=np.float64)

        H = B.copy()

        for _ in range(2, N + 1):
            numero_filtros = H.shape[1]

            filtros_suavizados = np.stack(
                [
                    np.convolve(H[:, k], B[:, 0], mode="full")
                    for k in range(numero_filtros)
                ],
                axis=1,
            )

            if numero_filtros <= D_1d:
                nuevo_detalle = np.convolve(
                    H[:, -1], B[:, 1], mode="full"
                )[:, None]
                H = np.concatenate([filtros_suavizados, nuevo_detalle], axis=1)
            else:
                H = filtros_suavizados

        # Misma normalización utilizada en dhtmtx.m:
        # C = 2^(N/2) * sqrt(H(1:D+1,1))
        constantes = (2.0 ** (N / 2.0)) * np.sqrt(H[: D_1d + 1, 0])
        H = H * constantes[None, :]

    # Comprobación de paridad: los órdenes pares son simétricos y los impares
    # antisimétricos. Esta validación detecta errores de construcción.
    for orden in range(H.shape[1]):
        error_paridad = np.max(
            np.abs(H[::-1, orden] - ((-1) ** orden) * H[:, orden])
        )
        if error_paridad > 1e-12:
            raise RuntimeError(
                f"El filtro de orden {orden} no cumple la paridad esperada."
            )

    if T is None:
        return H

    if N == 0:
        return H, H.copy()
    if not isinstance(T, (int, np.integer)) or not (1 <= T <= N):
        raise ValueError("T debe cumplir 1 <= T <= N.")

    if T <= 2:
        # Atajo exacto usado por idht2.m para T=1 y T=2.
        G = float(T) * H[::-1, :]
    else:
        # Caso general de dhtmtx.m. W corrige el solapamiento producido por
        # la malla de muestreo con separación T.
        W_conv = _matriz_convolucion(H[:, 0], N + 1)
        inicio = N - floor(N / T) * T
        filas = np.arange(inicio, 2 * N + 1, T)
        pesos = np.sum(W_conv[filas, :], axis=0)

        if np.any(np.isclose(pesos, 0.0)):
            raise ZeroDivisionError(
                "Se obtuvo un peso nulo al construir los filtros de síntesis."
            )
        G = H[::-1, :] / pesos[:, None]

    return H, G


# =============================================================================
# Transformada cartesiana directa
# =============================================================================


def _extender_simetrica_matlab(imagen: Array, N: int) -> Array:
    """Extensión simétrica equivalente al caso ``'symm'`` de ``dht2.m``."""
    borde = ceil(N / 2)
    if borde == 0:
        return imagen.copy()
    return np.pad(imagen, ((borde, borde), (borde, borde)), mode="symmetric")


def transformada_hermite_cartesiana(
    imagen: Array,
    N: int,
    D: int,
    T: int = 1,
    shape: str = "full",
) -> Coeficientes:
    """Calcula la DHT cartesiana bidimensional.

    Es el equivalente de ``dht2(X,N,D,T,shape)`` sin postprocesamiento.
    Los filtros son separables y la convolución para ``L_(m,n)`` utiliza el
    filtro de orden ``m`` en x y el de orden ``n`` en y.
    """
    shape = shape.lower()
    validar_parametros(N, D, T, shape)

    X = np.asarray(imagen, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("La transformada cartesiana requiere una imagen 2-D.")

    H = matriz_dht_binomial(N, D)

    if shape == "symm":
        X_filtrado = _extender_simetrica_matlab(X, N)
        modo_convolucion = "valid"
    else:
        X_filtrado = X
        modo_convolucion = shape

    coeficientes: Coeficientes = {}
    for orden_x, orden_y in ordenes_dht2(N, D):
        # np.outer crea primero el eje vertical (y) y después el horizontal (x).
        kernel_2d = np.outer(H[:, orden_y], H[:, orden_x])
        mapa = convolve2d(X_filtrado, kernel_2d, mode=modo_convolucion)
        coeficientes[(orden_x, orden_y)] = mapa[::T, ::T]

    return coeficientes


# =============================================================================
# Orientación y steering normalizado
# =============================================================================


def orientacion_gradiente(coeficientes: Mapping[Orden, Array]) -> Array:
    """Calcula ``theta = atan2(L_01, L_10)`` sin reducir módulo pi.

    En coordenadas de imagen, donde y aumenta hacia abajo, esta convención
    coincide con la orientación horaria indicada por ``rdht.m``.
    """
    if (1, 0) not in coeficientes or (0, 1) not in coeficientes:
        raise ValueError(
            "La orientación de gradiente requiere D >= 1 y los coeficientes "
            "L_(1,0) y L_(0,1)."
        )
    return np.arctan2(
        np.asarray(coeficientes[(0, 1)], dtype=np.float64),
        np.asarray(coeficientes[(1, 0)], dtype=np.float64),
    )


def _coeficientes_a_stack(
    coeficientes: Mapping[Orden, Array],
    ordenes: Sequence[Orden],
) -> Array:
    """Convierte un diccionario de coeficientes en un arreglo de canales."""
    faltantes = [orden for orden in ordenes if orden not in coeficientes]
    if faltantes:
        raise KeyError(f"Faltan coeficientes: {faltantes[:5]}")

    formas = {np.asarray(coeficientes[o]).shape for o in ordenes}
    if len(formas) != 1:
        raise ValueError("Todos los mapas de coeficientes deben tener la misma forma.")

    return np.stack(
        [np.asarray(coeficientes[o], dtype=np.float64) for o in ordenes],
        axis=-1,
    )


def _stack_a_coeficientes(stack: Array, ordenes: Sequence[Orden]) -> Coeficientes:
    """Convierte un stack de canales en un diccionario indexado por orden."""
    return {orden: stack[..., i].copy() for i, orden in enumerate(ordenes)}


def _angulo_como_mapa(
    theta: Union[float, Array],
    forma_espacial: Tuple[int, int],
) -> Array:
    """Convierte un ángulo escalar o un mapa angular a una matriz válida."""
    theta_array = np.asarray(theta, dtype=np.float64)
    if theta_array.ndim == 0:
        return np.full(forma_espacial, float(theta_array), dtype=np.float64)
    if theta_array.shape != forma_espacial:
        raise ValueError(
            f"theta tiene forma {theta_array.shape}; se esperaba {forma_espacial}."
        )
    return theta_array


def _rdht_stack(
    stack_cartesiano: Array,
    N: int,
    D: int,
    theta: Union[float, Array],
) -> Array:
    """Aplica la recurrencia normalizada de ``rdht.m`` a un stack.

    La recurrencia desnormaliza temporalmente los coeficientes interiores de
    cada bloque de orden total mediante ``sqrt(binom(q,k))``, aplica la
    rotación y vuelve a normalizarlos. Esto evita el error de usar directamente
    los coeficientes binomiales ordinarios sobre una base Hermite normalizada.
    """
    Y = np.asarray(stack_cartesiano, dtype=np.float64)
    if Y.ndim != 3:
        raise ValueError("El stack debe tener forma (filas, columnas, canales).")

    ordenes = ordenes_dht2(N, D)
    if Y.shape[-1] != len(ordenes):
        raise ValueError(
            f"Se esperaban {len(ordenes)} canales y se recibieron {Y.shape[-1]}."
        )

    angulo = _angulo_como_mapa(theta, Y.shape[:2])
    c = np.cos(angulo)
    s = np.sin(angulo)
    Z = Y.copy()

    posicion = 1  # El coeficiente L_00 no cambia con la rotación.

    # Bloques completos hasta la antidiagonal principal.
    for orden_total in range(1, min(D, N) + 1):
        longitud = orden_total + 1
        h = Y[..., posicion : posicion + longitud].copy()
        C = np.sqrt(
            np.array(
                [comb(orden_total, k) for k in range(longitud)],
                dtype=np.float64,
            )
        )

        if orden_total > 1:
            h[..., 1:orden_total] /= C[1:orden_total]

        longitud_h = longitud
        for indice_salida in range(orden_total):
            l = h.copy()
            longitud_l = longitud_h

            for _ in range(indice_salida, orden_total):
                l = (
                    c[..., None] * l[..., : longitud_l - 1]
                    + s[..., None] * l[..., 1:longitud_l]
                )
                longitud_l -= 1

            Z[..., posicion + indice_salida] = (
                l[..., 0] * C[indice_salida]
            )

            h = (
                c[..., None] * h[..., 1:longitud_h]
                - s[..., None] * h[..., : longitud_h - 1]
            )
            longitud_h -= 1

        Z[..., posicion + orden_total] = h[..., 0]
        posicion += longitud

    # Bloques por debajo de la antidiagonal, necesarios cuando D > N.
    for desplazamiento in range(1, min(D - N, N - 1) + 1):
        orden_reducido = N - desplazamiento
        longitud = orden_reducido + 1
        h = Y[..., posicion : posicion + longitud].copy()
        C = np.sqrt(
            np.array(
                [comb(orden_reducido, k) for k in range(longitud)],
                dtype=np.float64,
            )
        )

        if orden_reducido > 1:
            h[..., 1:orden_reducido] /= C[1:orden_reducido]

        longitud_h = longitud
        for indice_salida in range(orden_reducido):
            l = h.copy()
            longitud_l = longitud_h

            for _ in range(indice_salida, orden_reducido):
                l = (
                    c[..., None] * l[..., : longitud_l - 1]
                    + s[..., None] * l[..., 1:longitud_l]
                )
                longitud_l -= 1

            Z[..., posicion + indice_salida] = (
                l[..., 0] * C[indice_salida]
            )

            h = (
                c[..., None] * h[..., 1:longitud_h]
                - s[..., None] * h[..., : longitud_h - 1]
            )
            longitud_h -= 1

        Z[..., posicion + orden_reducido] = h[..., 0]
        posicion += longitud

    # En D=2N queda al final L_(N,N), bloque de un solo elemento. La misma
    # recurrencia de MATLAB lo deja invariante.
    canales_restantes = Y.shape[-1] - posicion
    if canales_restantes not in {0, 1}:
        raise RuntimeError(
            "La navegación de bloques de RDHT no cubrió correctamente los canales."
        )

    return Z


def rotar_coeficientes_hermite(
    coeficientes: Mapping[Orden, Array],
    N: int,
    D: int,
    theta: Union[float, Array],
) -> Coeficientes:
    """Convierte coeficientes cartesianos en coeficientes Hermite rotados."""
    ordenes = ordenes_dht2(N, D)
    stack = _coeficientes_a_stack(coeficientes, ordenes)
    stack_rotado = _rdht_stack(stack, N, D, theta)
    return _stack_a_coeficientes(stack_rotado, ordenes)


def desrotar_coeficientes_hermite(
    coeficientes_rotados: Mapping[Orden, Array],
    N: int,
    D: int,
    theta: Union[float, Array],
) -> Coeficientes:
    """Recupera los coeficientes cartesianos aplicando RDHT con ``-theta``."""
    return rotar_coeficientes_hermite(
        coeficientes_rotados,
        N=N,
        D=D,
        theta=-np.asarray(theta, dtype=np.float64),
    )


# =============================================================================
# Transformada inversa cartesiana
# =============================================================================


def _expandir_coeficientes_symm(
    stack: Array,
    forma_original: Tuple[int, int],
    N: int,
    T: int,
) -> Tuple[Array, Tuple[int, int], int]:
    """Reproduce la extensión de coeficientes del caso ``symm`` de idht2.m."""
    ysiz = (forma_original[0] + N, forma_original[1] + N)
    borde_izquierdo = floor(ceil(N / 2) / T)
    t0 = ceil(N / 2) - T * borde_izquierdo  # índice cero-base

    zsiz = (
        len(range(t0, ysiz[0], T)),
        len(range(t0, ysiz[1], T)),
    )
    borde_derecho = (
        zsiz[0] - stack.shape[0] - borde_izquierdo,
        zsiz[1] - stack.shape[1] - borde_izquierdo,
    )

    if min(*borde_derecho, borde_izquierdo) < 0:
        raise ValueError("Las dimensiones de los coeficientes no son compatibles con symm.")

    indices_filas = np.concatenate(
        [
            np.arange(borde_izquierdo - 1, -1, -1),
            np.arange(stack.shape[0]),
            np.arange(
                stack.shape[0] - 1,
                stack.shape[0] - borde_derecho[0] - 1,
                -1,
            ),
        ]
    ).astype(int)
    indices_columnas = np.concatenate(
        [
            np.arange(borde_izquierdo - 1, -1, -1),
            np.arange(stack.shape[1]),
            np.arange(
                stack.shape[1] - 1,
                stack.shape[1] - borde_derecho[1] - 1,
                -1,
            ),
        ]
    ).astype(int)

    expandido = np.zeros((zsiz[0], zsiz[1], stack.shape[2]), dtype=np.float64)

    # El low-pass se prolonga simétricamente.
    expandido[..., 0] = stack[..., 0][np.ix_(indices_filas, indices_columnas)]

    # Los demás coeficientes se insertan únicamente en la región central,
    # igual que en idht2.m.
    if stack.shape[2] > 1:
        fin_fila = zsiz[0] - borde_derecho[0]
        fin_columna = zsiz[1] - borde_derecho[1]
        expandido[
            borde_izquierdo:fin_fila,
            borde_izquierdo:fin_columna,
            1:,
        ] = stack[..., 1:]

    return expandido, ysiz, t0


def transformada_hermite_inversa(
    coeficientes: Mapping[Orden, Array],
    forma_original: Tuple[int, int],
    N: int,
    D: int,
    T: int = 1,
    shape: str = "full",
) -> Array:
    """Reconstruye una imagen desde coeficientes cartesianos.

    Traduce la etapa de síntesis de ``idht2.m``:

    1. Construye los filtros de interpolación ``G``.
    2. Inserta cada mapa sobre la malla espacial de paso ``T``.
    3. Convoluciona con el producto separable ``G_n(y) G_m(x)``.
    4. Suma las contribuciones de todos los órdenes disponibles.

    Con ``D < 2*N`` la reconstrucción es truncada. Con ``D=2*N``,
    ``shape='full'`` y los mismos parámetros de la transformada directa, la
    reconstrucción es perfecta salvo el error de punto flotante.
    """
    shape = shape.lower()
    validar_parametros(N, D, T, shape)

    if len(forma_original) != 2:
        raise ValueError("forma_original debe ser una tupla (alto, ancho).")

    ordenes = ordenes_dht2(N, D)
    stack = _coeficientes_a_stack(coeficientes, ordenes)
    _, G = matriz_dht_binomial(N, D, T)

    if shape == "full":
        ysiz = (forma_original[0] + N, forma_original[1] + N)
        modo_convolucion = "valid"
        t0 = 0
        stack_sintesis = stack
    elif shape == "same":
        ysiz = tuple(forma_original)
        modo_convolucion = "same"
        t0 = 0
        stack_sintesis = stack
    elif shape == "valid":
        ysiz = (forma_original[0] - N, forma_original[1] - N)
        if ysiz[0] <= 0 or ysiz[1] <= 0:
            raise ValueError("La imagen es demasiado pequeña para shape='valid'.")
        modo_convolucion = "full"
        t0 = 0
        stack_sintesis = stack
    else:  # symm
        stack_sintesis, ysiz, t0 = _expandir_coeficientes_symm(
            stack, forma_original, N, T
        )
        modo_convolucion = "valid"

    filas_malla = np.arange(t0, ysiz[0], T)
    columnas_malla = np.arange(t0, ysiz[1], T)
    forma_esperada = (len(filas_malla), len(columnas_malla))

    if stack_sintesis.shape[:2] != forma_esperada:
        raise ValueError(
            "Las dimensiones de los coeficientes no coinciden con N, T, shape "
            f"y la forma original. Se esperaba {forma_esperada} y se recibió "
            f"{stack_sintesis.shape[:2]}."
        )

    reconstruida: Optional[Array] = None

    for canal, (orden_x, orden_y) in enumerate(ordenes):
        mapa_expandido = np.zeros(ysiz, dtype=np.float64)
        mapa_expandido[np.ix_(filas_malla, columnas_malla)] = (
            stack_sintesis[..., canal]
        )

        kernel_sintesis = np.outer(G[:, orden_y], G[:, orden_x])
        contribucion = convolve2d(
            mapa_expandido,
            kernel_sintesis,
            mode=modo_convolucion,
        )

        if reconstruida is None:
            reconstruida = contribucion
        else:
            reconstruida += contribucion

    if reconstruida is None:
        raise RuntimeError("No se generó ninguna contribución de síntesis.")

    return reconstruida


# =============================================================================
# Métricas y salidas visuales
# =============================================================================


def metricas_reconstruccion(original: Array, reconstruida: Array) -> dict[str, float]:
    """Calcula MSE, RMSE, MAE, error máximo y PSNR sin aplicar clipping."""
    original = np.asarray(original, dtype=np.float64)
    reconstruida = np.asarray(reconstruida, dtype=np.float64)

    if original.shape != reconstruida.shape:
        raise ValueError(
            f"Las imágenes tienen formas distintas: {original.shape} y "
            f"{reconstruida.shape}."
        )

    diferencia = original - reconstruida
    mse = float(np.mean(diferencia**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diferencia)))
    max_abs = float(np.max(np.abs(diferencia)))
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs,
        "psnr_db": psnr,
    }


def metricas_ida_vuelta_coeficientes(
    originales: Mapping[Orden, Array],
    recuperados: Mapping[Orden, Array],
) -> dict[str, float]:
    """Mide el error de cartesianos -> rotados -> cartesianos."""
    if set(originales) != set(recuperados):
        raise ValueError("Los dos conjuntos de coeficientes deben tener los mismos órdenes.")

    diferencias = np.concatenate(
        [
            (
                np.asarray(originales[o], dtype=np.float64)
                - np.asarray(recuperados[o], dtype=np.float64)
            ).ravel()
            for o in originales
        ]
    )

    mse = float(np.mean(diferencias**2))
    return {
        "coefficient_mse": mse,
        "coefficient_rmse": float(np.sqrt(mse)),
        "coefficient_mae": float(np.mean(np.abs(diferencias))),
        "coefficient_max_abs_error": float(np.max(np.abs(diferencias))),
    }


def imagen_energia(
    coeficientes: Mapping[Orden, Array],
    incluir_dc: bool = False,
) -> Array:
    """Calcula la raíz de la energía cuadrática de los coeficientes."""
    primero = np.asarray(next(iter(coeficientes.values())), dtype=np.float64)
    energia = np.zeros_like(primero)

    for orden, mapa in coeficientes.items():
        if not incluir_dc and orden == (0, 0):
            continue
        energia += np.asarray(mapa, dtype=np.float64) ** 2

    return np.sqrt(energia)


def guardar_cuadricula_coeficientes(
    coeficientes: Mapping[Orden, Array],
    ruta: Ruta,
    titulo: str,
) -> None:
    """Guarda los coeficientes con m en columnas y n en filas."""
    salida = _preparar_ruta(ruta)
    ordenes = list(coeficientes)
    max_m = max(m for m, _ in ordenes)
    max_n = max(n for _, n in ordenes)

    figura, ejes = plt.subplots(
        max_n + 1,
        max_m + 1,
        figsize=(2.6 * (max_m + 1), 2.6 * (max_n + 1)),
        squeeze=False,
    )

    for n in range(max_n + 1):
        for m in range(max_m + 1):
            eje = ejes[n, m]
            orden = (m, n)

            if orden in coeficientes:
                mapa = np.asarray(coeficientes[orden], dtype=np.float64)
                limite = float(np.max(np.abs(mapa)))
                if limite == 0.0:
                    limite = 1.0

                # Escala simétrica: cero queda en gris medio, los valores
                # positivos y negativos conservan su signo visual.
                eje.imshow(mapa, cmap="gray", vmin=-limite, vmax=limite)
                eje.set_title(rf"$L_{{{m},{n}}}$")

            eje.axis("off")

    figura.suptitle(titulo, fontsize=15)
    figura.tight_layout()
    figura.savefig(salida, dpi=200, bbox_inches="tight")
    plt.close(figura)


def guardar_mapa_theta(theta: Array, ruta: Ruta) -> None:
    """Guarda el mapa de orientación en grados."""
    salida = _preparar_ruta(ruta)
    figura, eje = plt.subplots(figsize=(7, 6))
    grafica = eje.imshow(np.rad2deg(theta), cmap="gray")
    eje.set_title(r"Orientación local $\theta$ [grados]")
    eje.axis("off")
    figura.colorbar(grafica, ax=eje)
    figura.tight_layout()
    figura.savefig(salida, dpi=200, bbox_inches="tight")
    plt.close(figura)


def guardar_comparacion_reconstruccion(
    original: Array,
    reconstruida: Array,
    ruta: Ruta,
) -> None:
    """Guarda original, reconstrucción y error absoluto en una sola figura."""
    salida = _preparar_ruta(ruta)
    error = np.abs(original - reconstruida)

    figura, ejes = plt.subplots(1, 3, figsize=(14, 4.5))

    ejes[0].imshow(original, cmap="gray", vmin=0.0, vmax=1.0)
    ejes[0].set_title("Original")
    ejes[0].axis("off")

    # El clipping es únicamente para visualización. Las métricas se calculan
    # con la reconstrucción cruda.
    ejes[1].imshow(np.clip(reconstruida, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
    ejes[1].set_title("Reconstruida")
    ejes[1].axis("off")

    grafica_error = ejes[2].imshow(error, cmap="gray")
    ejes[2].set_title(r"Error absoluto $|I-\hat I|$")
    ejes[2].axis("off")
    figura.colorbar(grafica_error, ax=ejes[2], fraction=0.046, pad=0.04)

    figura.tight_layout()
    figura.savefig(salida, dpi=200, bbox_inches="tight")
    plt.close(figura)


def guardar_metricas_csv(
    grupos_metricas: Mapping[str, Mapping[str, float]],
    ruta: Ruta,
) -> None:
    """Guarda métricas organizadas por grupo en un CSV."""
    salida = _preparar_ruta(ruta)
    with salida.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["grupo", "metrica", "valor"])
        for grupo, metricas in grupos_metricas.items():
            for nombre, valor in metricas.items():
                escritor.writerow([grupo, nombre, valor])


# =============================================================================
# Función principal
# =============================================================================


def hermite_transform_image(
    image: ImagenEntrada,
    N: int = 8,
    D: int = 3,
    T: int = 1,
    shape: str = "full",
    use_rotation: bool = True,
    use_inverse_rotation: bool = False,
    use_inverse_transform: bool = False,
    rotation_mode: str = "gradient",
    angle: Union[float, Array] = 0.0,
    angle_unit: str = "degrees",
    output_paths: Optional[Mapping[str, Ruta]] = None,
) -> dict:
    """Ejecuta las etapas solicitadas de la transformada de Hermite.

    Parameters
    ----------
    image:
        Imagen de entrada.
    N:
        Escala discreta. Los filtros tienen longitud ``N+1``.
    D:
        Máximo orden total de la expansión. Se incluyen órdenes que cumplen
        ``m+n <= D`` y ``m,n <= N``. Para el cuadro completo use ``D=2*N``.
    T:
        Distancia de muestreo.
    shape:
        ``'full'`` (recomendado para validar reconstrucción), ``'same'``,
        ``'valid'`` o ``'symm'``.
    use_rotation:
        Si es ``True``, calcula los coeficientes rotados.
    use_inverse_rotation:
        Si es ``True``, aplica también la rotación inversa. Requiere
        ``use_rotation=True``.
    use_inverse_transform:
        Si es ``True``, reconstruye la imagen. Cuando hay rotación, la función
        deshace automáticamente la rotación antes de sintetizar.
    rotation_mode:
        ``'gradient'`` para orientación local o ``'fixed'`` para un ángulo
        proporcionado por ``angle``.
    angle:
        Ángulo escalar o mapa angular usado en modo ``'fixed'``.
    angle_unit:
        ``'degrees'`` o ``'radians'``.
    output_paths:
        Diccionario opcional. Claves admitidas:

        * ``cartesian_coefficients``
        * ``rotated_coefficients``
        * ``recovered_cartesian_coefficients``
        * ``theta``
        * ``reconstruction``
        * ``metrics_csv``

        Los directorios padres se crean automáticamente.

    Returns
    -------
    dict
        Imágenes, coeficientes, ángulo, métricas, filtros y metadatos.

    Examples
    --------
    Solo transformada cartesiana::

        resultado = hermite_transform_image(
            "house.tif", N=8, D=3, T=1,
            use_rotation=False,
            use_inverse_transform=False,
        )

    Flujo completo con reconstrucción perfecta::

        resultado = hermite_transform_image(
            "house.tif", N=8, D=16, T=1,
            use_rotation=True,
            use_inverse_rotation=True,
            use_inverse_transform=True,
            rotation_mode="gradient",
        )
    """
    shape = shape.lower()
    validar_parametros(N, D, T, shape)

    if use_inverse_rotation and not use_rotation:
        raise ValueError(
            "use_inverse_rotation=True requiere use_rotation=True. "
            "Para reconstruir directamente desde los cartesianos use "
            "use_inverse_transform=True."
        )

    X = leer_imagen_gris(image)
    rutas = dict(output_paths or {})
    ordenes = ordenes_dht2(N, D)

    # ------------------------------------------------------------------
    # 1) Imagen -> coeficientes cartesianos
    # ------------------------------------------------------------------
    coef_cartesianos = transformada_hermite_cartesiana(
        X, N=N, D=D, T=T, shape=shape
    )

    if "cartesian_coefficients" in rutas:
        guardar_cuadricula_coeficientes(
            coef_cartesianos,
            rutas["cartesian_coefficients"],
            f"DHT2 cartesiana: N={N}, D={D}, T={T}",
        )

    theta: Optional[Array] = None
    coef_rotados: Optional[Coeficientes] = None
    coef_cartesianos_recuperados: Optional[Coeficientes] = None
    metricas_coeficientes: Optional[dict[str, float]] = None

    # ------------------------------------------------------------------
    # 2) Coeficientes cartesianos -> coeficientes rotados
    # ------------------------------------------------------------------
    if use_rotation:
        modo = rotation_mode.lower()

        if modo in {"gradient", "grad", "dominant"}:
            theta = orientacion_gradiente(coef_cartesianos)
        elif modo in {"fixed", "angle"}:
            if angle_unit.lower() in {"degrees", "degree", "deg"}:
                theta = np.deg2rad(np.asarray(angle, dtype=np.float64))
            elif angle_unit.lower() in {"radians", "radian", "rad"}:
                theta = np.asarray(angle, dtype=np.float64)
            else:
                raise ValueError("angle_unit debe ser 'degrees' o 'radians'.")

            theta = _angulo_como_mapa(
                theta,
                next(iter(coef_cartesianos.values())).shape,
            )
        else:
            raise ValueError("rotation_mode debe ser 'gradient' o 'fixed'.")

        coef_rotados = rotar_coeficientes_hermite(
            coef_cartesianos, N=N, D=D, theta=theta
        )

        if "rotated_coefficients" in rutas:
            guardar_cuadricula_coeficientes(
                coef_rotados,
                rutas["rotated_coefficients"],
                f"DHT2 rotada: N={N}, D={D}, T={T}",
            )
        if "theta" in rutas:
            guardar_mapa_theta(theta, rutas["theta"])

    # ------------------------------------------------------------------
    # 3) Coeficientes rotados -> cartesianos recuperados
    # ------------------------------------------------------------------
    necesita_desrotar = use_rotation and (
        use_inverse_rotation or use_inverse_transform
    )

    if necesita_desrotar:
        assert coef_rotados is not None and theta is not None
        coef_cartesianos_recuperados = desrotar_coeficientes_hermite(
            coef_rotados, N=N, D=D, theta=theta
        )
        metricas_coeficientes = metricas_ida_vuelta_coeficientes(
            coef_cartesianos, coef_cartesianos_recuperados
        )

        if "recovered_cartesian_coefficients" in rutas:
            guardar_cuadricula_coeficientes(
                coef_cartesianos_recuperados,
                rutas["recovered_cartesian_coefficients"],
                "Coeficientes cartesianos recuperados",
            )

    # ------------------------------------------------------------------
    # 4) Coeficientes cartesianos -> imagen reconstruida
    # ------------------------------------------------------------------
    reconstruida: Optional[Array] = None
    metricas_imagen: Optional[dict[str, float]] = None

    if use_inverse_transform:
        coef_sintesis = (
            coef_cartesianos_recuperados
            if use_rotation
            else coef_cartesianos
        )
        assert coef_sintesis is not None

        reconstruida = transformada_hermite_inversa(
            coef_sintesis,
            forma_original=X.shape,
            N=N,
            D=D,
            T=T,
            shape=shape,
        )
        metricas_imagen = metricas_reconstruccion(X, reconstruida)

        if "reconstruction" in rutas:
            guardar_comparacion_reconstruccion(
                X, reconstruida, rutas["reconstruction"]
            )

    grupos_metricas: dict[str, Mapping[str, float]] = {}
    if metricas_coeficientes is not None:
        grupos_metricas["rotacion_ida_vuelta"] = metricas_coeficientes
    if metricas_imagen is not None:
        grupos_metricas["reconstruccion"] = metricas_imagen

    if "metrics_csv" in rutas and grupos_metricas:
        guardar_metricas_csv(grupos_metricas, rutas["metrics_csv"])

    # Coeficientes activos para visualización o uso en redes neuronales.
    coef_activos = coef_rotados if use_rotation else coef_cartesianos
    assert coef_activos is not None
    stack_activo = _coeficientes_a_stack(coef_activos, ordenes)

    H, G = matriz_dht_binomial(N, D, T)

    return {
        "original_image": X,
        "cartesian_coefficients": coef_cartesianos,
        "rotated_coefficients": coef_rotados,
        "recovered_cartesian_coefficients": coef_cartesianos_recuperados,
        "theta": theta,
        "reconstructed_image": reconstruida,
        "coefficient_roundtrip_metrics": metricas_coeficientes,
        "reconstruction_metrics": metricas_imagen,
        "active_coefficients": coef_activos,
        "coeff_stack": stack_activo,
        "transformed_image": imagen_energia(coef_activos),
        "orders": ordenes,
        "analysis_filters_1d": H,
        "synthesis_filters_1d": G,
        "parameters": {
            "N": N,
            "D": D,
            "T": T,
            "shape": shape,
            "filter_length": N + 1,
            "complete_square": D == 2 * N,
            "approximate_scale_s": N / 8.0,
            # En chtmtx.m: g(x)=exp(-x^2/(4s))/sqrt(4*pi*s), s=N/8.
            # Por tanto, la desviación estándar de esa gaussiana es sqrt(N)/2.
            "approximate_gaussian_std": np.sqrt(N) / 2.0,
        },
        "output_paths": rutas,
    }


# =============================================================================
# Ejemplo de uso
# =============================================================================


if __name__ == "__main__":
    ruta_ejemplo = Path("house.tif")

    if ruta_ejemplo.exists():
        # Para comparar los coeficientes con MATLAB use N=8, D=3, T=1.
        # Esta expansión es truncada, por lo que no se espera reconstrucción
        # perfecta. Para el cuadro completo cambie D a 2*N (=16).
        resultado = hermite_transform_image(
            image=ruta_ejemplo,
            N=8,
            D=3,
            T=1,
            shape="full",
            use_rotation=True,
            use_inverse_rotation=True,
            use_inverse_transform=True,
            rotation_mode="gradient",
            output_paths={
                "cartesian_coefficients": "resultados/cartesianos.png",
                "rotated_coefficients": "resultados/rotados.png",
                "recovered_cartesian_coefficients": "resultados/cartesianos_recuperados.png",
                "theta": "resultados/theta.png",
                "reconstruction": "resultados/reconstruccion.png",
                "metrics_csv": "resultados/metricas.csv",
            },
        )

        print("Órdenes:", resultado["orders"])
        print(
            "Error de rotación ida/vuelta:",
            resultado["coefficient_roundtrip_metrics"],
        )
        print("Métricas de reconstrucción:", resultado["reconstruction_metrics"])
    else:
        print(
            "Coloque house.tif junto a este script o importe "
            "hermite_transform_image desde otro archivo."
        )
