"""Transformada de Hermite 2-D con funciones Hermite--Gaussianas.

El modulo implementa analisis cartesiano, estimacion de orientacion,
``steering``, ``steering`` inverso y sintesis de una expansion local
truncada. Los filtros tienen soporte finito, impar y seleccionado
automaticamente; son productos separables de polinomios de Hermite
normalizados y una ventana Gaussiana.

La primera componente de un orden ``(m, n)`` corresponde al eje horizontal
``x`` (columnas), y la segunda al eje vertical ``y`` (filas).
"""

from __future__ import annotations

import csv
import warnings
from math import comb
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import convolve1d, correlate1d
from scipy.special import eval_hermite, gammaln

Array = np.ndarray
Order = Tuple[int, int]
CoeffDict = Dict[Order, Array]
PathLike = Union[str, Path]

_SUPPORT_TOLERANCE = 1e-8
_TAIL_SAMPLES = 3
_RGB_LUMINANCE_WEIGHTS = np.asarray(
    [0.298936021293775, 0.587043074451121, 0.114020904255103],
    dtype=np.float64,
)


# -----------------------------------------------------------------------------
# Image and path utilities
# -----------------------------------------------------------------------------


def _array_to_grayscale(image: Array, source: str) -> Array:
    """Convert a numeric image array to grayscale without changing its scale."""
    array = np.asarray(image)
    original_shape = array.shape
    if np.iscomplexobj(array):
        raise ValueError(
            f"{source} has complex values; a real grayscale image is required."
        )
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(
            f"{source} must contain numeric image data; received dtype {array.dtype}."
        )

    if array.ndim == 2:
        grayscale = array
    elif array.ndim == 3:
        channels = array.shape[-1]
        if channels in {1, 2}:
            # One channel is grayscale; two channels are luminance + alpha.
            grayscale = array[..., 0]
        elif channels in {3, 4}:
            # Alpha is display metadata and never enters the transform.
            rgb = array[..., :3].astype(np.float64, copy=False)
            grayscale = np.tensordot(
                rgb, _RGB_LUMINANCE_WEIGHTS, axes=([-1], [0])
            )
        else:
            raise ValueError(
                f"{source} has shape {original_shape}; expected 1, 2, 3 or 4 "
                "channels in the last dimension."
            )
    else:
        raise ValueError(
            f"{source} has shape {original_shape}; expected (height, width) "
            "or (height, width, channels)."
        )

    result = np.asarray(grayscale, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(
            f"{source} could not be converted to a 2-D image; got shape "
            f"{result.shape}."
        )
    if result.size == 0:
        raise ValueError(f"{source} is empty.")
    if not np.isfinite(result).all():
        raise ValueError(f"{source} contains NaN or infinite values.")
    return result


def _pil_to_grayscale(image: Image.Image, source: str) -> Array:
    """Apply EXIF orientation and convert a PIL image through the shared path."""
    oriented = ImageOps.exif_transpose(image)
    mode = oriented.mode
    direct_modes = {"1", "L", "LA", "I", "F", "RGB", "RGBA", "RGBX"}
    if mode in direct_modes or mode.startswith("I;16"):
        array = np.asarray(oriented)
    else:
        # Palette, CMYK, YCbCr and other encoded color spaces must first be
        # interpreted by PIL; the common RGB luminance formula is applied next.
        array = np.asarray(oriented.convert("RGB"))
    return _array_to_grayscale(array, f"{source} (PIL mode {mode!r})")


def read_image(image: Union[PathLike, Image.Image, Array]) -> Array:
    """Read an image as a two-dimensional grayscale array.

    Parameters
    ----------
    image : path-like, PIL.Image.Image or ndarray
        Input image. RGB arrays are converted with standard luminance weights.
    Returns
    -------
    image_array : ndarray
        Two-dimensional ``float64`` image in its original numeric scale.

    Notes
    -----
    Integer images are converted to ``float64`` without rescaling. In
    particular, an 8-bit file remains in ``0..255`` and a 16-bit grayscale
    TIFF is not reduced to 8 bits. RGB and RGBA inputs use the same luminance
    weights whether they come from a path, a PIL image or an array. Alpha is
    ignored. EXIF orientation is applied to PIL-backed inputs.
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as pil_image:
            return _pil_to_grayscale(pil_image, f"image file {str(image)!r}")
    elif isinstance(image, Image.Image):
        return _pil_to_grayscale(image, "PIL image")
    return _array_to_grayscale(image, "input array")


def _ensure_parent(path: Optional[PathLike]) -> Optional[Path]:
    if path is None:
        return None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a non-negative integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result


# -----------------------------------------------------------------------------
# Orders and Hermite--Gaussian filter bank
# -----------------------------------------------------------------------------


def hermite_orders(max_order: int, coefficient_region: str = "triangle") -> list[Order]:
    """Return coefficient orders grouped by total degree.

    Parameters
    ----------
    max_order : int
        Maximum total order for ``triangle`` or maximum order on each axis for
        ``square``.
    coefficient_region : {"triangle", "square"}, default="triangle"
        Region of coefficient pairs.

    Returns
    -------
    orders : list of tuple of int
        Channel order ``L00, L10, L01, L20, L11, L02, ...``. Within a total
        degree, ``m`` decreases and ``n`` increases.
    """
    limit = _nonnegative_integer("max_order", max_order)
    if not isinstance(coefficient_region, str):
        raise ValueError("coefficient_region must be 'triangle' or 'square'.")
    region = coefficient_region.lower()
    if region not in {"triangle", "square"}:
        raise ValueError("coefficient_region must be 'triangle' or 'square'.")

    max_total = limit if region == "triangle" else 2 * limit
    orders: list[Order] = []
    for total in range(max_total + 1):
        largest_m = total if region == "triangle" else min(limit, total)
        smallest_m = 0 if region == "triangle" else max(0, total - limit)
        orders.extend(
            (m, total - m)
            for m in range(largest_m, smallest_m - 1, -1)
        )
    return orders


def _hermite_normalization(order: int) -> float:
    """Return ``1 / sqrt(2**n * n!)`` without factorial overflow."""
    return float(
        np.exp(-0.5 * (order * np.log(2.0) + gammaln(order + 1.0)))
    )


def _analysis_filter_values(order: int, coordinates: Array, sigma: float) -> Array:
    """Evaluate the normalized one-dimensional analysis function."""
    scaled = np.asarray(coordinates, dtype=np.float64) / sigma
    gaussian = np.exp(-(scaled**2)) / (sigma * np.sqrt(np.pi))
    polynomial = _hermite_normalization(order) * eval_hermite(order, scaled)
    return np.asarray(polynomial * gaussian, dtype=np.float64)


def _select_support_radius(
    sigma: float,
    max_order: int,
    tolerance: float = _SUPPORT_TOLERANCE,
    tail_samples: int = _TAIL_SAMPLES,
) -> int:
    """Select an odd, symmetric finite support for all required 1-D filters.

    The search starts beyond both four Gaussian scales and the approximate
    turning point of the highest Hermite polynomial. A radius is accepted only
    when ``tail_samples`` consecutive non-negative samples of every filter are
    smaller than ``tolerance`` times that filter's sampled maximum.
    """
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be a finite positive number.")
    highest = _nonnegative_integer("max_order", max_order)
    if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be between zero and one.")
    if not isinstance(tail_samples, (int, np.integer)) or tail_samples < 1:
        raise ValueError("tail_samples must be a positive integer.")

    start_scale = max(4.0, np.sqrt(2.0 * highest + 1.0) + 2.0)
    radius = max(1, int(np.ceil(sigma * start_scale)))
    maximum_radius = max(radius + 1000, int(np.ceil(50.0 * sigma)) + highest)

    while radius <= maximum_radius:
        coordinates = np.arange(radius + tail_samples, dtype=np.float64)
        tails_are_small = True
        for order in range(highest + 1):
            values = np.abs(_analysis_filter_values(order, coordinates, sigma))
            peak = float(np.max(values))
            if peak == 0.0 or not np.all(values[radius:] / peak < tolerance):
                tails_are_small = False
                break
        if tails_are_small:
            return radius
        radius += 1

    raise RuntimeError(
        "Could not select a finite Hermite--Gaussian support for the requested "
        "sigma and order."
    )


def build_hermite_gaussian_filter_bank(
    max_order: int = 3,
    sigma: float = 2.0,
    coefficient_region: str = "square",
    dtype=np.float64,
) -> dict:
    """Build a finite-support Hermite--Gaussian analysis filter bank.

    Parameters
    ----------
    max_order : int, default=3
        Maximum total order in ``triangle`` mode or maximum order on each axis
        in ``square`` mode.
    sigma : float, default=2.0
        Gaussian scale in pixels.
    coefficient_region : {"triangle", "square"}, default="square"
        Public coefficient region.
    dtype : numpy dtype, default=float64
        Storage dtype. The filters are evaluated in ``float64`` first.

    Returns
    -------
    filter_bank : dict
        Contains public ``orders``, complete ``steering_orders``, 1-D and 2-D
        analysis filters, the Gaussian ``window_squared``, automatic support
        information and numerical diagnostics.

    Notes
    -----
    For ``square`` the bank also contains the complete triangular set through
    total order ``2*max_order``. Those auxiliary channels make every steering
    block complete; the public output remains square.
    """
    limit = _nonnegative_integer("max_order", max_order)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be a finite positive number.")
    sigma = float(sigma)
    public_orders = list(hermite_orders(limit, coefficient_region))
    region = coefficient_region.lower()
    steering_orders = (
        list(hermite_orders(2 * limit, "triangle"))
        if region == "square"
        else public_orders.copy()
    )
    highest_axis_order = max(
        (max(m, n) for m, n in steering_orders), default=0
    )
    support_radius = _select_support_radius(sigma, highest_axis_order)
    coordinates = np.arange(
        -support_radius, support_radius + 1, dtype=np.float64
    )

    filters_1d = {
        order: _analysis_filter_values(order, coordinates, sigma).astype(
            dtype, copy=False
        )
        for order in range(highest_axis_order + 1)
    }
    analysis_filters: CoeffDict = {
        (m, n): np.outer(filters_1d[n], filters_1d[m]).astype(dtype, copy=False)
        for m, n in steering_orders
    }
    window_squared = np.outer(filters_1d[0], filters_1d[0]).astype(
        dtype, copy=False
    )

    dc_leakage: dict[int, float] = {}
    diagnostics: dict[int, dict[str, float]] = {}
    dc_reference = max(abs(float(np.sum(filters_1d[0]))), np.finfo(float).eps)
    for order, values in filters_1d.items():
        values64 = np.asarray(values, dtype=np.float64)
        peak = max(float(np.max(np.abs(values64))), np.finfo(float).tiny)
        signed_sum = float(np.sum(values64))
        parity_error = float(
            np.max(np.abs(values64 - ((-1) ** order) * values64[::-1]))
        )
        edge_ratio = float(
            max(abs(values64[0]), abs(values64[-1])) / peak
        )
        relative_dc = abs(signed_sum) / dc_reference
        dc_leakage[order] = signed_sum
        diagnostics[order] = {
            "sum": signed_sum,
            "relative_dc_leakage": float(relative_dc),
            "parity_max_abs_error": parity_error,
            "edge_relative_magnitude": edge_ratio,
        }

    problematic = [
        order
        for order in range(1, highest_axis_order + 1)
        if diagnostics[order]["relative_dc_leakage"] > 1e-3
    ]
    if problematic:
        warnings.warn(
            "The sampled Hermite--Gaussian filters have appreciable DC "
            f"leakage at orders {problematic}. Consider a larger sigma relative "
            "to the pixel spacing.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "sigma": sigma,
        "support_radius": support_radius,
        "kernel_size": 2 * support_radius + 1,
        "coordinates": coordinates.astype(dtype, copy=False),
        "orders": public_orders,
        "steering_orders": steering_orders,
        "analysis_orders": steering_orders,
        "analysis_filters_1d": filters_1d,
        "analysis_filters": analysis_filters,
        "window": np.sqrt(window_squared).astype(dtype, copy=False),
        "window_squared": window_squared,
        "dc_leakage": dc_leakage,
        "diagnostics": diagnostics,
        "tail_tolerance": _SUPPORT_TOLERANCE,
        "tail_samples": _TAIL_SAMPLES,
        "coefficient_region": region,
        "max_order": limit,
    }


def build_hermite_filter_bank(
    max_order: int = 3,
    sigma: float = 2.0,
    coefficient_region: str = "square",
    dtype=np.float64,
) -> dict:
    """Compatibility name for :func:`build_hermite_gaussian_filter_bank`."""
    return build_hermite_gaussian_filter_bank(
        max_order=max_order,
        sigma=sigma,
        coefficient_region=coefficient_region,
        dtype=dtype,
    )


# -----------------------------------------------------------------------------
# Cartesian analysis and frame synthesis
# -----------------------------------------------------------------------------


def _normalize_boundary(boundary: str) -> str:
    if not isinstance(boundary, str):
        raise ValueError(
            "boundary must be one of: 'symmetric', 'constant', 'edge', or 'wrap'."
        )
    aliases = {
        "reflect": "symmetric",
        "symm": "symmetric",
        "symmetric": "symmetric",
        "constant": "constant",
        "fill": "constant",
        "zero": "constant",
        "edge": "edge",
        "replicate": "edge",
        "wrap": "wrap",
        "circular": "wrap",
    }
    try:
        return aliases[boundary.lower()]
    except KeyError as exc:
        raise ValueError(
            "boundary must be one of: 'symmetric', 'constant', 'edge', or 'wrap'."
        ) from exc


def _ndimage_boundary(boundary: str) -> str:
    """Map public boundary names to centered ``scipy.ndimage`` modes."""
    return {
        "symmetric": "reflect",
        "constant": "constant",
        "edge": "nearest",
        "wrap": "wrap",
    }[_normalize_boundary(boundary)]


def _sampling_positions(length: int, sampling_step: int) -> Array:
    return np.arange(0, length, sampling_step, dtype=int)


def cartesian_hermite_transform(
    image: Array,
    filter_bank: Mapping,
    sampling_step: int = 1,
    boundary: str = "symmetric",
    orders: Optional[Sequence[Order]] = None,
) -> CoeffDict:
    """Compute Cartesian Hermite coefficient maps by explicit correlation.

    Parameters
    ----------
    image : ndarray, shape (rows, columns)
        Input image in its original numeric scale.
    filter_bank : mapping
        Bank returned by :func:`build_hermite_gaussian_filter_bank`.
    sampling_step : int, default=1
        Distance in pixels between analysis positions.
    boundary : {"symmetric", "constant", "edge", "wrap"}, default="symmetric"
        Padding rule used before local correlation.
    orders : sequence of pairs, optional
        Channels to compute. By default the public bank orders are used.

    Returns
    -------
    coefficients : dict
        Two-dimensional maps indexed by ``(m, n)``.
    """
    if isinstance(sampling_step, (bool, np.bool_)) or not isinstance(
        sampling_step, (int, np.integer)
    ) or sampling_step < 1:
        raise ValueError("sampling_step must be a positive integer.")
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("image must be a non-empty two-dimensional array.")
    if not np.isfinite(values).all():
        raise ValueError("image contains NaN or infinite values.")

    selected_orders = list(filter_bank["orders"] if orders is None else orders)
    missing = [
        order
        for order in selected_orders
        if order not in filter_bank["analysis_filters"]
    ]
    if missing:
        raise KeyError(f"The filter bank does not contain orders {missing[:5]}.")

    mode = _ndimage_boundary(boundary)
    horizontal_orders = sorted({m for m, _ in selected_orders})
    horizontal_responses = {
        m: correlate1d(
            values,
            np.asarray(filter_bank["analysis_filters_1d"][m], dtype=np.float64),
            axis=1,
            mode=mode,
            cval=0.0,
        )
        for m in horizontal_orders
    }
    coefficients: CoeffDict = {}
    for m, n in selected_orders:
        dense = correlate1d(
            horizontal_responses[m],
            np.asarray(filter_bank["analysis_filters_1d"][n], dtype=np.float64),
            axis=0,
            mode=mode,
            cval=0.0,
        )
        coefficients[(m, n)] = dense[::sampling_step, ::sampling_step].copy()
    return coefficients


def synthesize_hermite_image(
    coefficients: Mapping[Order, Array],
    filter_bank: Mapping,
    image_shape: Tuple[int, int],
    sampling_step: int = 1,
    boundary: str = "symmetric",
) -> Array:
    """Synthesize an image by overlap-add and frame normalization.

    Parameters
    ----------
    coefficients : mapping
        Cartesian coefficient maps for the public orders of ``filter_bank``.
    filter_bank : mapping
        Hermite--Gaussian filter bank used in analysis.
    image_shape : tuple of int
        Requested output shape.
    sampling_step : int, default=1
        Analysis-grid spacing.
    boundary : {"symmetric", "constant", "edge", "wrap"}, default="symmetric"
        Must match analysis. It fixes the coefficient convention; overlap-add
        uses the same centered support and sampling positions.

    Returns
    -------
    reconstructed : ndarray
        Truncated frame synthesis with exactly ``image_shape``.

    Notes
    -----
    The numerator sums shifted analysis functions weighted by their local
    coefficients. The denominator is the sum of shifted Gaussian windows
    squared over exactly the same sampling lattice.
    """
    if isinstance(sampling_step, (bool, np.bool_)) or not isinstance(
        sampling_step, (int, np.integer)
    ) or sampling_step < 1:
        raise ValueError("sampling_step must be a positive integer.")
    _normalize_boundary(boundary)
    if (
        not isinstance(image_shape, tuple)
        or len(image_shape) != 2
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
            for value in image_shape
        )
    ):
        raise ValueError("image_shape must contain two positive integers.")

    height, width = (int(image_shape[0]), int(image_shape[1]))
    orders = list(filter_bank["orders"])
    missing = [order for order in orders if order not in coefficients]
    if missing:
        raise KeyError(f"Missing coefficients required for synthesis: {missing[:5]}.")

    row_positions = _sampling_positions(height, sampling_step)
    column_positions = _sampling_positions(width, sampling_step)
    expected_shape = (len(row_positions), len(column_positions))
    for order in orders:
        if np.asarray(coefficients[order]).shape != expected_shape:
            raise ValueError(
                f"Coefficient L_{order} has shape "
                f"{np.asarray(coefficients[order]).shape}; expected {expected_shape}."
            )

    sampling_mask = np.zeros((height, width), dtype=np.float64)
    sampling_mask[np.ix_(row_positions, column_positions)] = 1.0
    numerator = np.zeros((height, width), dtype=np.float64)

    for m, n in orders:
        upsampled = np.zeros((height, width), dtype=np.float64)
        upsampled[np.ix_(row_positions, column_positions)] = np.asarray(
            coefficients[(m, n)], dtype=np.float64
        )
        horizontal = convolve1d(
            upsampled,
            np.asarray(filter_bank["analysis_filters_1d"][m], dtype=np.float64),
            axis=1,
            mode="constant",
            cval=0.0,
        )
        numerator += convolve1d(
            horizontal,
            np.asarray(filter_bank["analysis_filters_1d"][n], dtype=np.float64),
            axis=0,
            mode="constant",
            cval=0.0,
        )

    denominator = convolve1d(
        sampling_mask,
        np.asarray(filter_bank["analysis_filters_1d"][0], dtype=np.float64),
        axis=1,
        mode="constant",
        cval=0.0,
    )
    denominator = convolve1d(
        denominator,
        np.asarray(filter_bank["analysis_filters_1d"][0], dtype=np.float64),
        axis=0,
        mode="constant",
        cval=0.0,
    )
    if np.any(denominator <= 0.0):
        raise ValueError(
            "sampling_step leaves pixels without coverage for the automatically "
            "selected finite support."
        )
    return np.divide(numerator, denominator).astype(np.float64, copy=False)


def inverse_cartesian_hermite_transform(
    coefficients: Mapping[Order, Array],
    filter_bank: Mapping,
    image_shape: Tuple[int, int],
    sampling_step: int = 1,
    boundary: str = "symmetric",
) -> Array:
    """Compatibility name for :func:`synthesize_hermite_image`."""
    return synthesize_hermite_image(
        coefficients,
        filter_bank,
        image_shape,
        sampling_step=sampling_step,
        boundary=boundary,
    )


# -----------------------------------------------------------------------------
# Orientation and steering
# -----------------------------------------------------------------------------


def dominant_gradient_theta(coefficients: Mapping[Order, Array]) -> Array:
    """Return ``atan2(L_01, L_10)`` in radians without modulo reduction."""
    if (1, 0) not in coefficients or (0, 1) not in coefficients:
        raise ValueError(
            "Dominant gradient orientation requires coefficients (1, 0) and (0, 1)."
        )
    return np.arctan2(
        np.asarray(coefficients[(0, 1)], dtype=np.float64),
        np.asarray(coefficients[(1, 0)], dtype=np.float64),
    )


def _coefficients_to_stack(
    coefficients: Mapping[Order, Array], orders: Sequence[Order]
) -> Array:
    missing = [order for order in orders if order not in coefficients]
    if missing:
        raise KeyError(f"Missing coefficient orders: {missing[:5]}.")
    values = [np.asarray(coefficients[order], dtype=np.float64) for order in orders]
    if not values:
        raise ValueError("At least one coefficient order is required.")
    if any(value.shape != values[0].shape for value in values):
        raise ValueError("All coefficient maps must have the same shape.")
    if values[0].ndim != 2:
        raise ValueError("Coefficient maps must be two-dimensional.")
    return np.stack(values, axis=-1)


def _stack_to_coefficients(stack: Array, orders: Sequence[Order]) -> CoeffDict:
    return {
        order: np.asarray(stack[..., channel], dtype=np.float64).copy()
        for channel, order in enumerate(orders)
    }


def _rotation_angle_array(theta: Union[float, Array], spatial_shape: Tuple[int, int]) -> Array:
    angle = np.asarray(theta, dtype=np.float64)
    if not np.isfinite(angle).all():
        raise ValueError("theta must contain only finite values.")
    if angle.ndim == 0:
        return np.full(spatial_shape, float(angle), dtype=np.float64)
    try:
        return np.broadcast_to(angle, spatial_shape).astype(np.float64, copy=False)
    except ValueError as exc:
        raise ValueError(
            f"theta has shape {angle.shape}; expected a scalar or a map "
            f"broadcastable to {spatial_shape}."
        ) from exc


def _rotate_complete_block(block: Array, theta: Array) -> Array:
    """Rotate one complete normalized Hermite block of equal total order."""
    degree = block.shape[-1] - 1
    if degree <= 0:
        return np.asarray(block, dtype=np.float64).copy()

    cosine = np.cos(theta)
    sine = np.sin(theta)
    normalization = np.sqrt(
        np.asarray([comb(degree, index) for index in range(degree + 1)])
    )
    work = np.asarray(block, dtype=np.float64).copy()
    if degree > 1:
        work[..., 1:degree] /= normalization[1:degree]

    rotated = np.empty_like(work)
    active_length = degree + 1
    for output_order in range(degree):
        reduced = work.copy()
        reduced_length = active_length
        for _ in range(output_order, degree):
            reduced = (
                cosine[..., None] * reduced[..., : reduced_length - 1]
                + sine[..., None] * reduced[..., 1:reduced_length]
            )
            reduced_length -= 1
        rotated[..., output_order] = (
            reduced[..., 0] * normalization[output_order]
        )
        work = (
            cosine[..., None] * work[..., 1:active_length]
            - sine[..., None] * work[..., : active_length - 1]
        )
        active_length -= 1
    rotated[..., degree] = work[..., 0]
    return rotated


def _rotate_complete_coefficients(
    coefficients: Mapping[Order, Array],
    theta: Union[float, Array],
    orders: Sequence[Order],
) -> CoeffDict:
    stack = _coefficients_to_stack(coefficients, orders)
    angle_map = _rotation_angle_array(theta, stack.shape[:2])
    result = stack.copy()
    for total in sorted({m + n for m, n in orders}):
        block_orders = [(m, total - m) for m in range(total, -1, -1)]
        missing = [order for order in block_orders if order not in orders]
        if missing:
            raise ValueError(
                f"Steering requires the complete total-order block {total}; "
                f"missing {missing}."
            )
        indices = [orders.index(order) for order in block_orders]
        result[..., indices] = _rotate_complete_block(
            stack[..., indices], angle_map
        )
    return _stack_to_coefficients(result, orders)


def rotate_hermite_coefficients(
    coefficients: Mapping[Order, Array],
    theta: Union[float, Array],
    max_order: int,
    coefficient_region: str = "square",
) -> CoeffDict:
    """Rotate coefficients within complete blocks of equal total order.

    Parameters
    ----------
    coefficients : mapping
        Cartesian coefficient maps. In ``square`` mode this mapping must also
        contain the auxiliary triangle through total order ``2*max_order``.
    theta : float or ndarray
        Steering angle in radians.
    max_order : int
        Region limit.
    coefficient_region : {"triangle", "square"}, default="square"
        Determines the complete order set required for steering.

    Returns
    -------
    rotated : dict
        Complete rotated mapping. A square caller can extract its public pairs
        after steering.
    """
    limit = _nonnegative_integer("max_order", max_order)
    if not isinstance(coefficient_region, str):
        raise ValueError("coefficient_region must be 'triangle' or 'square'.")
    region = coefficient_region.lower()
    if region == "triangle":
        required_orders = list(hermite_orders(limit, "triangle"))
    elif region == "square":
        required_orders = list(hermite_orders(2 * limit, "triangle"))
    else:
        raise ValueError("coefficient_region must be 'triangle' or 'square'.")
    missing = [order for order in required_orders if order not in coefficients]
    if missing:
        raise ValueError(
            "Square steering requires auxiliary complete blocks through total "
            f"order {2 * limit}; missing {missing[:5]}."
            if region == "square"
            else f"Steering coefficients are missing orders {missing[:5]}."
        )
    return _rotate_complete_coefficients(coefficients, theta, required_orders)


def inverse_rotate_hermite_coefficients(
    rotated_coefficients: Mapping[Order, Array],
    theta: Union[float, Array],
    max_order: int,
    coefficient_region: str = "square",
) -> CoeffDict:
    """Undo steering by applying the same complete-block operator at ``-theta``."""
    return rotate_hermite_coefficients(
        rotated_coefficients,
        -np.asarray(theta, dtype=np.float64),
        max_order=max_order,
        coefficient_region=coefficient_region,
    )


def _extract_coefficients(
    coefficients: Mapping[Order, Array], orders: Sequence[Order]
) -> CoeffDict:
    return {order: np.asarray(coefficients[order]) for order in orders}


# -----------------------------------------------------------------------------
# Metrics and visual outputs
# -----------------------------------------------------------------------------


def coefficient_roundtrip_metrics(
    original: Mapping[Order, Array], recovered: Mapping[Order, Array]
) -> dict:
    """Measure the numerical error of steering followed by inverse steering."""
    orders = list(original.keys())
    if not orders:
        raise ValueError("At least one coefficient map is required.")
    missing = [order for order in orders if order not in recovered]
    if missing:
        raise KeyError(f"Recovered coefficients are missing orders {missing[:5]}.")
    differences = np.concatenate(
        [
            (
                np.asarray(original[order], dtype=np.float64)
                - np.asarray(recovered[order], dtype=np.float64)
            ).ravel()
            for order in orders
        ]
    )
    mse = float(np.mean(differences**2))
    return {
        "coefficient_mse": mse,
        "coefficient_rmse": float(np.sqrt(mse)),
        "coefficient_mae": float(np.mean(np.abs(differences))),
        "coefficient_max_abs_error": float(np.max(np.abs(differences))),
    }


def reconstruction_metrics(
    original: Array,
    reconstructed: Array,
    data_range: Optional[float] = None,
) -> dict:
    """Compute MSE, RMSE, MAE, maximum error and PSNR."""
    source = np.asarray(original)
    estimate = np.asarray(reconstructed)
    if source.shape != estimate.shape or source.size == 0:
        raise ValueError("original and reconstructed must have the same non-empty shape.")
    source64 = source.astype(np.float64, copy=False)
    estimate64 = estimate.astype(np.float64, copy=False)
    difference = source64 - estimate64
    absolute = np.abs(difference)
    mse = float(np.mean(difference**2))
    rmse = float(np.sqrt(mse))

    if data_range is None:
        if np.issubdtype(source.dtype, np.integer):
            limits = np.iinfo(source.dtype)
            dynamic_range = float(limits.max - limits.min)
        else:
            dynamic_range = float(np.max(source64) - np.min(source64))
            if dynamic_range == 0.0:
                dynamic_range = max(float(np.max(np.abs(source64))), 1.0)
    else:
        dynamic_range = float(data_range)
        if not np.isfinite(dynamic_range) or dynamic_range <= 0.0:
            raise ValueError("data_range must be finite and positive.")
    psnr = (
        float("inf")
        if mse == 0.0
        else float(20.0 * np.log10(dynamic_range / rmse))
    )
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": float(np.mean(absolute)),
        "max_abs_error": float(np.max(absolute)),
        "psnr": psnr,
    }


def energy_image(
    coefficients: Mapping[Order, Array], include_dc: bool = False
) -> Array:
    """Return the root-sum-square energy of selected coefficient maps."""
    if not coefficients:
        raise ValueError("At least one coefficient map is required.")
    first = np.asarray(next(iter(coefficients.values())), dtype=np.float64)
    energy = np.zeros_like(first)
    for order, value in coefficients.items():
        if not include_dc and order == (0, 0):
            continue
        energy += np.asarray(value, dtype=np.float64) ** 2
    return np.sqrt(energy)


def save_coefficients_grid(
    coefficients: Mapping[Order, Array],
    output_path: PathLike,
    title: str,
    *,
    cmap: str = "gray",
) -> None:
    """Save coefficient maps in their ``(m, n)`` positions.

    Parameters
    ----------
    coefficients : mapping
        Scalar coefficient maps indexed by ``(m, n)``.
    output_path : path-like
        Destination image path.
    title : str
        Figure title.
    cmap : str, default="gray"
        Matplotlib colormap. The symmetric display limits place zero at the
        midpoint of the grayscale range.

    Notes
    -----
    The colormap and display limits affect only the rendered figure. They never
    modify the coefficient arrays used by analysis, steering or synthesis.
    """
    if not coefficients:
        raise ValueError("At least one coefficient map is required.")
    orders = list(coefficients.keys())
    if any(
        not isinstance(order, tuple)
        or len(order) != 2
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 0
            for value in order
        )
        for order in orders
    ):
        raise ValueError("Coefficient keys must be non-negative integer pairs (m, n).")
    display_values = {}
    for order in orders:
        value = np.asarray(coefficients[order])
        if value.ndim != 2 or value.size == 0 or np.iscomplexobj(value):
            raise ValueError(
                f"Coefficient L_{order} must be a non-empty, real 2-D array."
            )
        if not np.isfinite(value).all():
            raise ValueError(
                f"Coefficient L_{order} contains NaN or infinite values."
            )
        display_values[order] = value

    path = _ensure_parent(output_path)
    max_m = max(m for m, _ in orders)
    max_n = max(n for _, n in orders)
    figure, axes = plt.subplots(
        max_n + 1,
        max_m + 1,
        figsize=(2.7 * (max_m + 1), 2.7 * (max_n + 1)),
        squeeze=False,
    )
    for n in range(max_n + 1):
        for m in range(max_m + 1):
            axis = axes[n, m]
            order = (m, n)
            if order in display_values:
                value = display_values[order]
                limit = max(float(np.max(np.abs(value))), np.finfo(float).eps)
                axis.imshow(value, cmap=cmap, vmin=-limit, vmax=limit)
                axis.set_title(rf"$L_{{{m},{n}}}$")
            axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_theta_image(theta: Array, output_path: PathLike) -> None:
    """Save an orientation map in degrees with a cyclic angular colormap."""
    angle = np.asarray(theta)
    if angle.ndim != 2 or angle.size == 0 or np.iscomplexobj(angle):
        raise ValueError("theta must be a non-empty, real two-dimensional map.")
    if not np.isfinite(angle).all():
        raise ValueError("theta contains NaN or infinite values.")
    path = _ensure_parent(output_path)
    figure, axis = plt.subplots(figsize=(7, 6))
    shown = axis.imshow(np.rad2deg(angle), cmap="twilight", vmin=-180, vmax=180)
    axis.set_title(r"Orientacion local $\theta$ [grados]")
    axis.axis("off")
    figure.colorbar(shown, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_intensity_image(
    image: Array,
    output_path: PathLike,
    title: str,
) -> None:
    """Save a two-dimensional scalar image using explicit grayscale display.

    Parameters
    ----------
    image : ndarray, shape (rows, columns)
        Intensity image or non-negative scalar summary.
    output_path : path-like
        Destination image path.
    title : str
        Figure title.

    Notes
    -----
    Rendering does not normalize or mutate ``image``. Matplotlib maps its
    numeric range to grayscale only in the saved visualization.
    """
    values = np.asarray(image)
    if values.ndim != 2 or values.size == 0 or np.iscomplexobj(values):
        raise ValueError("image must be a non-empty, real two-dimensional array.")
    if not np.isfinite(values).all():
        raise ValueError("image contains NaN or infinite values.")

    path = _ensure_parent(output_path)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if lower == upper:
        upper = lower + 1.0
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(values, cmap="gray", vmin=lower, vmax=upper)
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_reconstruction_comparison(
    original: Array, reconstructed: Array, output_path: PathLike
) -> None:
    """Save original, reconstructed image and absolute error."""
    source = np.asarray(original)
    estimate = np.asarray(reconstructed)
    if (
        source.ndim != 2
        or estimate.ndim != 2
        or source.size == 0
        or source.shape != estimate.shape
        or np.iscomplexobj(source)
        or np.iscomplexobj(estimate)
    ):
        raise ValueError(
            "original and reconstructed must be non-empty, real 2-D arrays "
            "with the same shape."
        )
    if not np.isfinite(source).all() or not np.isfinite(estimate).all():
        raise ValueError("original and reconstructed must contain finite values.")
    path = _ensure_parent(output_path)
    original64 = source.astype(np.float64, copy=False)
    reconstructed64 = estimate.astype(np.float64, copy=False)
    error = np.abs(original64 - reconstructed64)
    lower = float(min(np.min(original64), np.min(reconstructed64)))
    upper = float(max(np.max(original64), np.max(reconstructed64)))
    if lower == upper:
        upper = lower + 1.0

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(original64, cmap="gray", vmin=lower, vmax=upper)
    axes[0].set_title("Original")
    axes[1].imshow(reconstructed64, cmap="gray", vmin=lower, vmax=upper)
    axes[1].set_title("Reconstruida")
    shown_error = axes[2].imshow(error, cmap="gray", vmin=0.0)
    axes[2].set_title(r"Error absoluto $|I-\hat I|$")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(shown_error, ax=axes[2], fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_metrics_csv(metrics: Mapping[str, float], output_path: PathLike) -> None:
    """Save scalar metrics as a two-column CSV file."""
    path = _ensure_parent(output_path)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------


def hermite_transform_image(
    image: Union[PathLike, Image.Image, Array],
    max_order: int = 3,
    sigma: float = 2.0,
    coefficient_region: str = "square",
    sampling_step: int = 1,
    boundary: str = "symmetric",
    use_rotation: bool = True,
    use_inverse_rotation: bool = True,
    use_inverse_transform: bool = True,
    rotation_mode: str = "dominant",
    angle: Union[float, Array] = 0.0,
    angle_unit: str = "degrees",
    output_paths: Optional[Mapping[str, PathLike]] = None,
) -> dict:
    """Run the finite-support Hermite--Gaussian transform workflow.

    Parameters
    ----------
    image : path-like, PIL.Image.Image or ndarray
        Grayscale or RGB input. Its numeric scale is preserved.
    max_order : int, default=3
        Maximum total degree for ``triangle`` or maximum per-axis degree for
        ``square``.
    sigma : float, default=2.0
        Gaussian scale in pixels. Filter support is selected automatically.
    coefficient_region : {"triangle", "square"}, default="square"
        Visible coefficient set.
    sampling_step : int, default=1
        Pixel spacing of the local analysis grid.
    boundary : {"symmetric", "constant", "edge", "wrap"}, default="symmetric"
        Boundary extension used by analysis.
    use_rotation : bool, default=True
        Estimate/use ``theta`` and steer the Cartesian coefficients.
    use_inverse_rotation : bool, default=True
        Recover Cartesian coefficients after steering.
    use_inverse_transform : bool, default=True
        Synthesize an image from Cartesian coefficients. When rotation is
        active, inverse steering is performed before synthesis.
    rotation_mode : {"dominant", "fixed"}, default="dominant"
        ``dominant`` uses ``atan2(L_01, L_10)``; ``fixed`` uses ``angle``.
    angle : float or ndarray, default=0.0
        Fixed scalar angle or map broadcastable to coefficient-map shape.
    angle_unit : {"degrees", "radians"}, default="degrees"
        Unit of ``angle`` in fixed mode.
    output_paths : mapping, optional
        Paths keyed by ``original_image``, ``cartesian_coefficients``,
        ``rotated_coefficients``, ``recovered_cartesian_coefficients``,
        ``theta``, ``reconstruction``, ``reconstructed_image``,
        ``coefficient_energy``, ``transformed_image`` or ``metrics_csv``.

    Returns
    -------
    result : dict
        Public coefficient maps, angle, optional reconstruction and metrics,
        channel stack, energy summary, filter diagnostics and output paths.

    Notes
    -----
    Synthesis uses only the requested finite coefficient region and is
    therefore a truncated approximation. In square mode, complete auxiliary
    blocks through total degree ``2*max_order`` are used internally for
    steering and inverse steering, then only public square pairs are returned.
    """
    if use_inverse_rotation and not use_rotation:
        raise ValueError("use_inverse_rotation=True requires use_rotation=True.")
    if not all(
        isinstance(flag, (bool, np.bool_))
        for flag in (use_rotation, use_inverse_rotation, use_inverse_transform)
    ):
        raise ValueError("Workflow flags must be boolean values.")

    image_array = read_image(image)
    paths = dict(output_paths or {})
    if "original_image" in paths:
        save_intensity_image(
            image_array,
            paths["original_image"],
            "Imagen original en escala de grises",
        )
    filter_bank = build_hermite_gaussian_filter_bank(
        max_order=max_order,
        sigma=sigma,
        coefficient_region=coefficient_region,
        dtype=np.float64,
    )
    public_orders = list(filter_bank["orders"])
    steering_orders = list(filter_bank["steering_orders"])
    analysis_orders = steering_orders if use_rotation else public_orders

    full_cartesian = cartesian_hermite_transform(
        image_array,
        filter_bank,
        sampling_step=sampling_step,
        boundary=boundary,
        orders=analysis_orders,
    )
    cartesian_coefficients = _extract_coefficients(full_cartesian, public_orders)
    if "cartesian_coefficients" in paths:
        save_coefficients_grid(
            cartesian_coefficients,
            paths["cartesian_coefficients"],
            "Coeficientes Hermite cartesianos",
        )

    theta = None
    full_rotated = None
    rotated_coefficients = None
    full_recovered = None
    recovered_cartesian_coefficients = None
    coefficient_metrics = None

    if use_rotation:
        if not isinstance(rotation_mode, str):
            raise ValueError("rotation_mode must be 'dominant' or 'fixed'.")
        mode = rotation_mode.lower()
        if mode in {"dominant", "gradient", "grad"}:
            theta = dominant_gradient_theta(full_cartesian)
        elif mode in {"fixed", "angle"}:
            if not isinstance(angle_unit, str):
                raise ValueError("angle_unit must be 'degrees' or 'radians'.")
            unit = angle_unit.lower()
            if unit in {"degree", "degrees", "deg"}:
                theta = np.deg2rad(np.asarray(angle, dtype=np.float64))
            elif unit in {"radian", "radians", "rad"}:
                theta = np.asarray(angle, dtype=np.float64)
            else:
                raise ValueError("angle_unit must be 'degrees' or 'radians'.")
        else:
            raise ValueError("rotation_mode must be 'dominant' or 'fixed'.")

        full_rotated = rotate_hermite_coefficients(
            full_cartesian,
            theta,
            max_order=max_order,
            coefficient_region=coefficient_region,
        )
        rotated_coefficients = _extract_coefficients(full_rotated, public_orders)
        if "rotated_coefficients" in paths:
            save_coefficients_grid(
                rotated_coefficients,
                paths["rotated_coefficients"],
                "Coeficientes Hermite rotados",
            )
        if "theta" in paths:
            save_theta_image(
                _rotation_angle_array(
                    theta, next(iter(cartesian_coefficients.values())).shape
                ),
                paths["theta"],
            )

    need_inverse_rotation = use_rotation and (
        use_inverse_rotation or use_inverse_transform
    )
    if need_inverse_rotation:
        full_recovered = inverse_rotate_hermite_coefficients(
            full_rotated,
            theta,
            max_order=max_order,
            coefficient_region=coefficient_region,
        )
        recovered_cartesian_coefficients = _extract_coefficients(
            full_recovered, public_orders
        )
        coefficient_metrics = coefficient_roundtrip_metrics(
            cartesian_coefficients, recovered_cartesian_coefficients
        )
        if "recovered_cartesian_coefficients" in paths:
            save_coefficients_grid(
                recovered_cartesian_coefficients,
                paths["recovered_cartesian_coefficients"],
                "Coeficientes cartesianos recuperados",
            )

    reconstructed_image = None
    image_metrics = None
    if use_inverse_transform:
        synthesis_coefficients = (
            recovered_cartesian_coefficients
            if use_rotation
            else cartesian_coefficients
        )
        reconstructed_image = synthesize_hermite_image(
            synthesis_coefficients,
            filter_bank,
            image_shape=image_array.shape,
            sampling_step=sampling_step,
            boundary=boundary,
        )
        image_metrics = reconstruction_metrics(image_array, reconstructed_image)
        if "reconstruction" in paths:
            save_reconstruction_comparison(
                image_array, reconstructed_image, paths["reconstruction"]
            )
        if "reconstructed_image" in paths:
            save_intensity_image(
                reconstructed_image,
                paths["reconstructed_image"],
                "Imagen reconstruida",
            )

    all_metrics: dict[str, float] = {}
    if coefficient_metrics is not None:
        all_metrics.update(coefficient_metrics)
    if image_metrics is not None:
        all_metrics.update(image_metrics)
    if "metrics_csv" in paths and all_metrics:
        save_metrics_csv(all_metrics, paths["metrics_csv"])

    active_coefficients = (
        rotated_coefficients if use_rotation else cartesian_coefficients
    )
    active_stack = _coefficients_to_stack(active_coefficients, public_orders)
    coefficient_energy = energy_image(active_coefficients)
    if "coefficient_energy" in paths:
        save_intensity_image(
            coefficient_energy,
            paths["coefficient_energy"],
            "Energia de coeficientes Hermite",
        )
    if "transformed_image" in paths:
        save_intensity_image(
            coefficient_energy,
            paths["transformed_image"],
            "Energia de coeficientes Hermite",
        )

    return {
        "original_image": image_array,
        "cartesian_coefficients": cartesian_coefficients,
        "rotated_coefficients": rotated_coefficients,
        "recovered_cartesian_coefficients": recovered_cartesian_coefficients,
        "theta": theta,
        "reconstructed_image": reconstructed_image,
        "coefficient_roundtrip_metrics": coefficient_metrics,
        "reconstruction_metrics": image_metrics,
        "active_coefficients": active_coefficients,
        "coeff_stack": active_stack,
        "coefficient_energy": coefficient_energy,
        "transformed_image": coefficient_energy,
        "orders": public_orders,
        "auxiliary_orders": steering_orders,
        "filter_bank": filter_bank,
        "support_radius": filter_bank["support_radius"],
        "output_paths": paths,
    }


if __name__ == "__main__":
    result = hermite_transform_image(
        image=Path("Fusion_Images_ds") / "house.tif",
        max_order=3,
        sigma=2.0,
        coefficient_region="square",
        sampling_step=1,
        use_rotation=True,
        use_inverse_rotation=True,
        use_inverse_transform=True,
        rotation_mode="dominant",
        output_paths={
            "original_image": "results/original_grayscale.png",
            "cartesian_coefficients": "results/cartesian_coefficients.png",
            "rotated_coefficients": "results/rotated_coefficients.png",
            "recovered_cartesian_coefficients": "results/recovered_cartesian.png",
            "theta": "results/theta.png",
            "reconstruction": "results/reconstruction.png",
            "reconstructed_image": "results/reconstructed_grayscale.png",
            "coefficient_energy": "results/coefficient_energy.png",
            "metrics_csv": "results/metrics.csv",
        },
    )
    print("Orders:", result["orders"])
    print("Support radius:", result["support_radius"])
    print("Coefficient round-trip:", result["coefficient_roundtrip_metrics"])
    print("Reconstruction:", result["reconstruction_metrics"])
