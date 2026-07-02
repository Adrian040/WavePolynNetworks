"""Hermite transform, steering, inverse steering and image reconstruction.

This implementation keeps the continuous Hermite-Gaussian approach of the
original Python script, but corrects the main consistency issues:

1. Analysis, steering, inverse steering and synthesis are separate functions.
2. Steering uses the normalized recurrence used by the MATLAB RDHT toolbox.
3. The dominant gradient angle is kept in [-pi, pi] (no modulo-pi reduction).
4. The angle is returned separately; no coefficient channel is overwritten.
5. Reconstruction uses discrete dual synthesis filters computed from the
   sampled analysis filters. With the full square basis and
   kernel_size == max_order + 1, the local transform is complete and the
   reconstruction is numerically exact, up to floating-point precision.
6. All requested stages can be selected from hermite_transform_image().

The full-square exact mode is a hybrid: the analysis filters are sampled
continuous Hermite-Gaussian functions, while the separate dual synthesis
filters follow the same analysis/synthesis philosophy as the MATLAB toolbox.
"""

from __future__ import annotations

import csv
from math import comb
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from scipy.special import eval_hermite, gammaln
from numpy.lib.stride_tricks import sliding_window_view

Array = np.ndarray
Order = Tuple[int, int]
CoeffDict = Dict[Order, Array]
PathLike = Union[str, Path]


# -----------------------------------------------------------------------------
# Image and path utilities
# -----------------------------------------------------------------------------


def read_image(image: Union[PathLike, Image.Image, Array], dtype=np.float64) -> Array:
    """Read an image as a two-dimensional grayscale array in [0, 1]."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as pil_image:
            img = np.asarray(pil_image.convert("L"), dtype=dtype)
    elif isinstance(image, Image.Image):
        img = np.asarray(image.convert("L"), dtype=dtype)
    else:
        img = np.asarray(image)
        if img.ndim == 3:
            if img.shape[-1] >= 3:
                # Standard luminance conversion instead of a simple channel mean.
                rgb = img[..., :3].astype(dtype)
                img = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            elif img.shape[-1] == 2:
                # Grayscale + alpha: keep the grayscale channel.
                img = img[..., 0]
            else:
                img = np.squeeze(img, axis=-1)
        if img.ndim != 2:
            raise ValueError(f"The input image must be 2-D after conversion; got shape {img.shape}.")
        img = img.astype(dtype, copy=False)

    if img.size == 0:
        raise ValueError("The input image is empty.")

    max_value = float(np.nanmax(img))
    if max_value > 1.5:
        img = img / 255.0

    if not np.isfinite(img).all():
        raise ValueError("The input image contains NaN or infinite values.")

    return img.astype(dtype, copy=False)


def _ensure_parent(path: Optional[PathLike]) -> Optional[Path]:
    if path is None:
        return None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


# -----------------------------------------------------------------------------
# Orders and filter construction
# -----------------------------------------------------------------------------


def hermite_orders(max_order: int, coefficient_region: str = "triangle") -> Sequence[Order]:
    """Return coefficient orders in the same anti-diagonal order as MATLAB DHTORD.

    Parameters
    ----------
    max_order:
        In ``triangle`` mode, it is the maximum total order m+n.
        In ``square`` mode, it is the maximum order on each axis.
    coefficient_region:
        ``triangle`` -> m+n <= max_order.
        ``square``   -> 0 <= m,n <= max_order.
    """
    if not isinstance(max_order, (int, np.integer)) or max_order < 0:
        raise ValueError("max_order must be a non-negative integer.")

    region = coefficient_region.lower()
    if region == "triangle":
        n_scale = max_order
        max_total = max_order
    elif region == "square":
        n_scale = max_order
        max_total = 2 * max_order
    else:
        raise ValueError("coefficient_region must be 'triangle' or 'square'.")

    orders = []
    for total in range(max_total + 1):
        m_min = max(0, total - n_scale)
        m_max = min(n_scale, total)
        for m in range(m_min, m_max + 1):
            orders.append((m, total - m))
    return orders


def _hermite_normalization(order: int) -> float:
    """Return 1/sqrt(2^n n!) without overflowing for moderate orders."""
    return float(np.exp(-0.5 * (order * np.log(2.0) + gammaln(order + 1.0))))


def build_hermite_filter_bank(
    max_order: int = 3,
    sigma: float = 2.0,
    kernel_size: Optional[int] = None,
    coefficient_region: str = "square",
    exact_reconstruction: bool = True,
    rcond: float = 1e-12,
    dtype=np.float64,
) -> dict:
    """Build sampled Hermite-Gaussian analysis filters and their discrete duals.

    The analysis filter is

        D_mn(x,y) = G_mn(x,y) * w(x,y)^2,

    where G_mn contains the normalized physicists' Hermite polynomials and
    w is an isotropic Gaussian window.

    The synthesis filters are the discrete dual basis obtained from the
    pseudoinverse of the sampled analysis operator. In full-square mode with
    ``kernel_size == max_order + 1``, the operator is square and full-rank,
    yielding numerical perfect reconstruction of every local patch.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if rcond <= 0:
        raise ValueError("rcond must be positive.")

    region = coefficient_region.lower()
    orders = list(hermite_orders(max_order, region))

    if exact_reconstruction:
        if region != "square":
            raise ValueError(
                "exact_reconstruction=True requires coefficient_region='square'."
            )
        required_kernel = max_order + 1
        if kernel_size is None:
            kernel_size = required_kernel
        elif kernel_size != required_kernel:
            raise ValueError(
                "For exact reconstruction with the continuous sampled basis, "
                f"kernel_size must equal max_order + 1 = {required_kernel}."
            )
    elif kernel_size is None:
        kernel_size = int(2 * np.ceil(3.0 * sigma) + 1)

    if not isinstance(kernel_size, (int, np.integer)) or kernel_size <= 0:
        raise ValueError("kernel_size must be a positive integer.")

    # Half-integer coordinates are allowed for an even-size kernel, matching
    # the N+1 support convention of the MATLAB discrete transform.
    coords = np.arange(kernel_size, dtype=dtype) - (kernel_size - 1.0) / 2.0
    x_scaled = coords / float(sigma)

    max_axis_order = max(max(m, n) for m, n in orders) if orders else 0
    hermite_1d = {}
    for order in range(max_axis_order + 1):
        hermite_1d[order] = (
            _hermite_normalization(order) * eval_hermite(order, x_scaled)
        ).astype(dtype)

    yy, xx = np.meshgrid(x_scaled, x_scaled, indexing="ij")
    window = np.exp(-0.5 * (xx**2 + yy**2)).astype(dtype)
    window_squared = (window**2).astype(dtype)

    analysis_filters: CoeffDict = {}
    for m, n in orders:
        # First index m is the x order; second index n is the y order.
        polynomial = np.outer(hermite_1d[n], hermite_1d[m]).astype(dtype)
        analysis_filters[(m, n)] = (polynomial * window_squared).astype(dtype)

    analysis_matrix = np.column_stack(
        [analysis_filters[order].reshape(-1) for order in orders]
    ).astype(dtype)

    # c = A^T p, therefore p = pinv(A^T)c.
    synthesis_matrix = np.linalg.pinv(analysis_matrix.T, rcond=rcond).astype(dtype)
    synthesis_filters: CoeffDict = {
        order: synthesis_matrix[:, index].reshape(kernel_size, kernel_size)
        for index, order in enumerate(orders)
    }

    rank = int(np.linalg.matrix_rank(analysis_matrix, tol=rcond))
    condition_number = float(np.linalg.cond(analysis_matrix))
    patch_dimension = kernel_size * kernel_size
    is_complete = len(orders) == patch_dimension and rank == patch_dimension

    if exact_reconstruction and not is_complete:
        raise np.linalg.LinAlgError(
            "The sampled Hermite analysis matrix is not full rank; exact "
            "reconstruction cannot be guaranteed with these parameters."
        )

    return {
        "orders": orders,
        "analysis_filters": analysis_filters,
        "synthesis_filters": synthesis_filters,
        "window": window,
        "window_squared": window_squared,
        "analysis_matrix": analysis_matrix,
        "synthesis_matrix": synthesis_matrix,
        "kernel_size": int(kernel_size),
        "sigma": float(sigma),
        "rank": rank,
        "condition_number": condition_number,
        "is_complete": is_complete,
        "coefficient_region": region,
        "max_order": int(max_order),
    }


# -----------------------------------------------------------------------------
# Cartesian forward transform and inverse transform
# -----------------------------------------------------------------------------


def _padding_for_kernel(kernel_size: int) -> Tuple[int, int, int, int]:
    top = kernel_size // 2
    bottom = kernel_size - 1 - top
    left = kernel_size // 2
    right = kernel_size - 1 - left
    return top, bottom, left, right


def _pad_image(image: Array, pads: Tuple[int, int, int, int], boundary: str) -> Array:
    top, bottom, left, right = pads
    boundary = boundary.lower()
    pad_width = ((top, bottom), (left, right))

    if boundary in {"reflect", "symm", "symmetric"}:
        # np.pad('symmetric') repeats the edge sample and is closest to
        # scipy.signal boundary='symm'.
        return np.pad(image, pad_width, mode="symmetric")
    if boundary in {"constant", "fill", "zero"}:
        return np.pad(image, pad_width, mode="constant", constant_values=0.0)
    if boundary in {"edge", "replicate"}:
        return np.pad(image, pad_width, mode="edge")
    if boundary in {"wrap", "circular"}:
        return np.pad(image, pad_width, mode="wrap")
    raise ValueError(
        "boundary must be one of: 'symmetric', 'constant', 'edge', or 'wrap'."
    )


def cartesian_hermite_transform(
    image: Array,
    filter_bank: Mapping,
    sampling_step: int = 1,
    boundary: str = "symmetric",
) -> CoeffDict:
    """Compute dense or subsampled Cartesian Hermite coefficient maps."""
    if not isinstance(sampling_step, (int, np.integer)) or sampling_step < 1:
        raise ValueError("sampling_step must be a positive integer.")

    kernel_size = int(filter_bank["kernel_size"])
    max_sampling_step = max(1, int(filter_bank["max_order"]))
    if sampling_step > max_sampling_step:
        raise ValueError(
            "sampling_step must satisfy T <= max_order, matching the MATLAB "
            f"DHT restriction. Received T={sampling_step}, max_order={filter_bank['max_order']}."
        )

    orders = list(filter_bank["orders"])
    filters = np.stack(
        [filter_bank["analysis_filters"][order] for order in orders], axis=0
    )

    pads = _padding_for_kernel(kernel_size)
    padded = _pad_image(image, pads, boundary)
    patch_view = sliding_window_view(padded, (kernel_size, kernel_size))
    sampled_patches = patch_view[::sampling_step, ::sampling_step]

    # Correlation: no spatial reversal of the analysis filter.
    coefficient_stack = np.einsum(
        "ijxy,kxy->ijk", sampled_patches, filters, optimize=True
    )

    return {
        order: coefficient_stack[..., index].copy()
        for index, order in enumerate(orders)
    }


def inverse_cartesian_hermite_transform(
    coefficients: Mapping[Order, Array],
    filter_bank: Mapping,
    image_shape: Tuple[int, int],
    sampling_step: int = 1,
    boundary: str = "symmetric",
) -> Array:
    """Reconstruct an image by dual-filter synthesis and overlap-add."""
    if not isinstance(sampling_step, (int, np.integer)) or sampling_step < 1:
        raise ValueError("sampling_step must be a positive integer.")

    orders = list(filter_bank["orders"])
    missing = [order for order in orders if order not in coefficients]
    if missing:
        raise KeyError(f"Missing coefficients required for synthesis: {missing[:5]}")

    first_shape = np.asarray(coefficients[orders[0]]).shape
    if len(first_shape) != 2:
        raise ValueError("Each coefficient map must be two-dimensional.")
    for order in orders:
        if np.asarray(coefficients[order]).shape != first_shape:
            raise ValueError("All coefficient maps must have the same shape.")

    height, width = image_shape
    kernel_size = int(filter_bank["kernel_size"])
    pads = _padding_for_kernel(kernel_size)

    # The coefficient at (i,j) represents the patch whose top-left location
    # in the padded image is (i*T,j*T).
    position_height = height
    position_width = width
    accumulator_shape = (
        position_height + kernel_size - 1,
        position_width + kernel_size - 1,
    )
    accumulator = np.zeros(accumulator_shape, dtype=np.float64)

    sampling_mask = np.zeros((position_height, position_width), dtype=np.float64)
    row_positions = np.arange(0, height, sampling_step)
    col_positions = np.arange(0, width, sampling_step)

    expected_shape = (len(row_positions), len(col_positions))
    if first_shape != expected_shape:
        raise ValueError(
            f"Coefficient maps have shape {first_shape}, but {expected_shape} "
            "is expected from image_shape and sampling_step."
        )

    sampling_mask[np.ix_(row_positions, col_positions)] = 1.0

    for order in orders:
        upsampled = np.zeros((position_height, position_width), dtype=np.float64)
        upsampled[np.ix_(row_positions, col_positions)] = coefficients[order]
        synthesis_filter = filter_bank["synthesis_filters"][order]
        accumulator += convolve2d(upsampled, synthesis_filter, mode="full")

    coverage = convolve2d(
        sampling_mask, np.ones((kernel_size, kernel_size), dtype=np.float64), mode="full"
    )
    reconstructed_padded = np.divide(
        accumulator,
        coverage,
        out=np.zeros_like(accumulator),
        where=coverage > 0,
    )
    top, bottom, left, right = pads
    cropped_coverage = coverage[
        top : top + height,
        left : left + width,
    ]
    if np.any(cropped_coverage <= 0):
        raise RuntimeError(
            "The chosen sampling_step leaves uncovered pixels inside the output image."
        )
    reconstructed = reconstructed_padded[
        top : top + height,
        left : left + width,
    ]
    return reconstructed.astype(np.float64, copy=False)


# -----------------------------------------------------------------------------
# Orientation and normalized steering (ported from MATLAB RDHT recurrence)
# -----------------------------------------------------------------------------


def dominant_gradient_theta(coefficients: Mapping[Order, Array]) -> Array:
    """Return theta = atan2(L_01, L_10) in [-pi, pi], without modulo pi."""
    if (1, 0) not in coefficients or (0, 1) not in coefficients:
        raise ValueError(
            "Dominant gradient orientation requires coefficients (1,0) and (0,1)."
        )
    return np.arctan2(coefficients[(0, 1)], coefficients[(1, 0)]).astype(np.float64)


def _coefficients_to_stack(
    coefficients: Mapping[Order, Array], orders: Sequence[Order]
) -> Array:
    missing = [order for order in orders if order not in coefficients]
    if missing:
        raise KeyError(f"Missing coefficient orders: {missing[:5]}")
    return np.stack([np.asarray(coefficients[order]) for order in orders], axis=-1)


def _stack_to_coefficients(stack: Array, orders: Sequence[Order]) -> CoeffDict:
    return {order: stack[..., index].copy() for index, order in enumerate(orders)}


def _rotation_angle_array(theta: Union[float, Array], spatial_shape: Tuple[int, int]) -> Array:
    angle = np.asarray(theta, dtype=np.float64)
    if angle.ndim == 0:
        return np.full(spatial_shape, float(angle), dtype=np.float64)
    if angle.shape != spatial_shape:
        raise ValueError(
            f"theta has shape {angle.shape}; expected scalar or {spatial_shape}."
        )
    return angle


def _rdht_forward_stack(
    coefficient_stack: Array,
    theta: Union[float, Array],
    max_order: int,
    max_total_order: int,
) -> Array:
    """MATLAB-compatible normalized RDHT recurrence, without angle embedding."""
    y = np.asarray(coefficient_stack, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError("coefficient_stack must have shape (rows, cols, channels).")

    expected_orders = []
    for total in range(max_total_order + 1):
        for m in range(max(0, total - max_order), min(max_order, total) + 1):
            expected_orders.append((m, total - m))
    if y.shape[-1] != len(expected_orders):
        raise ValueError(
            f"Expected {len(expected_orders)} coefficient channels; got {y.shape[-1]}."
        )

    angle = _rotation_angle_array(theta, y.shape[:2])
    c = np.cos(angle)
    s = np.sin(angle)
    z = y.copy()

    position = 1  # skip L_00

    # Complete total-order blocks above/on the main anti-diagonal.
    for total in range(1, min(max_total_order, max_order) + 1):
        block_length = total + 1
        h = y[..., position : position + block_length].copy()
        normalization = np.sqrt(
            np.array([comb(total, k) for k in range(block_length)], dtype=np.float64)
        )

        if total > 1:
            h[..., 1:total] /= normalization[1:total]

        h_length = block_length
        for m_index in range(total):
            l = h.copy()
            l_length = h_length
            for _ in range(m_index, total):
                l = (
                    c[..., None] * l[..., : l_length - 1]
                    + s[..., None] * l[..., 1:l_length]
                )
                l_length -= 1

            z[..., position + m_index] = l[..., 0] * normalization[m_index]
            h = (
                c[..., None] * h[..., 1:h_length]
                - s[..., None] * h[..., : h_length - 1]
            )
            h_length -= 1

        z[..., position + total] = h[..., 0]
        position += block_length

    # Partial blocks below the main anti-diagonal. This is the part needed
    # for a full square m,n <= max_order when max_total_order > max_order.
    extra_blocks = min(max_total_order - max_order, max_order - 1)
    for offset in range(1, extra_blocks + 1):
        reduced_order = max_order - offset
        block_length = reduced_order + 1
        h = y[..., position : position + block_length].copy()
        normalization = np.sqrt(
            np.array(
                [comb(reduced_order, k) for k in range(block_length)],
                dtype=np.float64,
            )
        )

        if reduced_order > 1:
            h[..., 1:reduced_order] /= normalization[1:reduced_order]

        h_length = block_length
        for m_index in range(reduced_order):
            l = h.copy()
            l_length = h_length
            for _ in range(m_index, reduced_order):
                l = (
                    c[..., None] * l[..., : l_length - 1]
                    + s[..., None] * l[..., 1:l_length]
                )
                l_length -= 1

            z[..., position + m_index] = l[..., 0] * normalization[m_index]
            h = (
                c[..., None] * h[..., 1:h_length]
                - s[..., None] * h[..., : h_length - 1]
            )
            h_length -= 1

        z[..., position + reduced_order] = h[..., 0]
        position += block_length

    # For the full square D=2N, the final corner coefficient L_{N,N}
    # forms a one-element block and is invariant under this finite RDHT
    # recurrence, so it remains unchanged in z.
    if position not in {y.shape[-1], y.shape[-1] - 1}:
        raise RuntimeError(
            f"RDHT channel traversal ended at {position}, but {y.shape[-1]} channels exist."
        )

    return z


def rotate_hermite_coefficients(
    coefficients: Mapping[Order, Array],
    theta: Union[float, Array],
    max_order: int,
    coefficient_region: str = "square",
) -> CoeffDict:
    """Rotate normalized Hermite coefficients with the MATLAB RDHT recurrence."""
    orders = list(hermite_orders(max_order, coefficient_region))
    max_total_order = max(m + n for m, n in orders) if orders else 0
    stack = _coefficients_to_stack(coefficients, orders)
    rotated = _rdht_forward_stack(stack, theta, max_order, max_total_order)
    return _stack_to_coefficients(rotated, orders)


def inverse_rotate_hermite_coefficients(
    rotated_coefficients: Mapping[Order, Array],
    theta: Union[float, Array],
    max_order: int,
    coefficient_region: str = "square",
) -> CoeffDict:
    """Undo RDHT steering by applying the same recurrence at angle -theta."""
    angle = -np.asarray(theta, dtype=np.float64)
    return rotate_hermite_coefficients(
        rotated_coefficients,
        angle,
        max_order=max_order,
        coefficient_region=coefficient_region,
    )


# -----------------------------------------------------------------------------
# Metrics and visual outputs
# -----------------------------------------------------------------------------


def coefficient_roundtrip_metrics(
    original: Mapping[Order, Array], recovered: Mapping[Order, Array]
) -> dict:
    orders = list(original.keys())
    differences = np.concatenate(
        [(np.asarray(original[o]) - np.asarray(recovered[o])).ravel() for o in orders]
    )
    mse = float(np.mean(differences**2))
    return {
        "coefficient_mse": mse,
        "coefficient_rmse": float(np.sqrt(mse)),
        "coefficient_mae": float(np.mean(np.abs(differences))),
        "coefficient_max_abs_error": float(np.max(np.abs(differences))),
    }


def reconstruction_metrics(original: Array, reconstructed: Array) -> dict:
    difference = np.asarray(original, dtype=np.float64) - np.asarray(
        reconstructed, dtype=np.float64
    )
    mse = float(np.mean(difference**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(difference)))
    max_abs = float(np.max(np.abs(difference)))
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs,
        "psnr": psnr,
    }


def energy_image(coefficients: Mapping[Order, Array], include_dc: bool = False) -> Array:
    first = next(iter(coefficients.values()))
    energy = np.zeros_like(first, dtype=np.float64)
    for order, value in coefficients.items():
        if not include_dc and order == (0, 0):
            continue
        energy += np.asarray(value, dtype=np.float64) ** 2
    return np.sqrt(energy)


def save_coefficients_grid(
    coefficients: Mapping[Order, Array],
    output_path: PathLike,
    title: str,
) -> None:
    path = _ensure_parent(output_path)
    orders = list(coefficients.keys())
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
            if order in coefficients:
                value = np.asarray(coefficients[order])
                limit = float(np.max(np.abs(value)))
                if limit == 0.0:
                    limit = 1.0
                axis.imshow(value, cmap="gray", vmin=-limit, vmax=limit)
                axis.set_title(rf"$L_{{{m},{n}}}$")
            axis.axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_theta_image(theta: Array, output_path: PathLike) -> None:
    path = _ensure_parent(output_path)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(np.rad2deg(theta), cmap="gray")
    axis.set_title(r"Orientación local $\theta$ [grados]")
    axis.axis("off")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_reconstruction_comparison(
    original: Array,
    reconstructed: Array,
    output_path: PathLike,
) -> None:
    path = _ensure_parent(output_path)
    error = np.abs(original - reconstructed)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(original, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Display clipping is visual only; metrics use the raw reconstruction.
    axes[1].imshow(np.clip(reconstructed, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Reconstruida")
    axes[1].axis("off")

    error_plot = axes[2].imshow(error, cmap="gray")
    axes[2].set_title(r"Error absoluto $|I-\hat I|$")
    axes[2].axis("off")
    figure.colorbar(error_plot, ax=axes[2], fraction=0.046, pad=0.04)

    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_metrics_csv(metrics: Mapping[str, float], output_path: PathLike) -> None:
    path = _ensure_parent(output_path)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


# -----------------------------------------------------------------------------
# Main orchestration function
# -----------------------------------------------------------------------------


def hermite_transform_image(
    image: Union[PathLike, Image.Image, Array],
    max_order: int = 3,
    sigma: float = 2.0,
    kernel_size: Optional[int] = None,
    coefficient_region: str = "square",
    exact_reconstruction: bool = True,
    sampling_step: int = 1,
    boundary: str = "symmetric",
    use_rotation: bool = True,
    use_inverse_rotation: bool = False,
    use_inverse_transform: bool = False,
    rotation_mode: str = "dominant",
    angle: Union[float, Array] = 0.0,
    angle_unit: str = "degrees",
    output_paths: Optional[Mapping[str, PathLike]] = None,
    rcond: float = 1e-12,
) -> dict:
    """Run any requested subset of the Cartesian/rotated/inverse HT workflow.

    Stage selection
    ---------------
    ``use_rotation=False, use_inverse_transform=False``
        Cartesian transform only.

    ``use_rotation=True, use_inverse_transform=False``
        Cartesian transform followed by rotation.

    ``use_rotation=True, use_inverse_rotation=True,
    use_inverse_transform=False``
        Cartesian transform -> rotation -> inverse rotation. No image synthesis.

    ``use_rotation=False, use_inverse_transform=True``
        Cartesian transform followed by direct Cartesian synthesis.

    ``use_rotation=True, use_inverse_transform=True``
        Full flow: Cartesian transform -> rotation -> inverse rotation -> image
        synthesis. Inverse rotation is performed automatically because it is
        required before Cartesian synthesis.

    Parameters
    ----------
    max_order:
        In ``square`` mode, maximum order independently on x and y. The full
        coefficient set contains (max_order+1)^2 maps.
        In ``triangle`` mode, maximum total order m+n.
    exact_reconstruction:
        Requires ``square`` mode and ``kernel_size=max_order+1``. This creates
        a complete local discrete basis and dual synthesis filters.
    use_inverse_rotation:
        Enables the rotation round trip even when image synthesis is disabled.
    output_paths:
        Optional paths with any of these keys:
        ``cartesian_coefficients``, ``rotated_coefficients``,
        ``recovered_cartesian_coefficients``, ``theta``, ``reconstruction``,
        ``metrics_csv``. Parent directories are created automatically.
    """
    if use_inverse_rotation and not use_rotation:
        raise ValueError(
            "use_inverse_rotation=True requires use_rotation=True. "
            "For Cartesian image reconstruction without rotation, use "
            "use_inverse_transform=True instead."
        )

    image_array = read_image(image, dtype=np.float64)
    paths = dict(output_paths or {})

    filter_bank = build_hermite_filter_bank(
        max_order=max_order,
        sigma=sigma,
        kernel_size=kernel_size,
        coefficient_region=coefficient_region,
        exact_reconstruction=exact_reconstruction,
        rcond=rcond,
        dtype=np.float64,
    )

    # 1) Image -> Cartesian coefficients.
    cartesian_coefficients = cartesian_hermite_transform(
        image_array,
        filter_bank,
        sampling_step=sampling_step,
        boundary=boundary,
    )

    if "cartesian_coefficients" in paths:
        save_coefficients_grid(
            cartesian_coefficients,
            paths["cartesian_coefficients"],
            "Coeficientes Hermite cartesianos",
        )

    theta = None
    rotated_coefficients = None
    recovered_cartesian_coefficients = None
    coefficient_metrics = None

    # 2) Cartesian -> rotated coefficients.
    if use_rotation:
        mode = rotation_mode.lower()
        if mode in {"dominant", "gradient", "grad"}:
            theta = dominant_gradient_theta(cartesian_coefficients)
        elif mode in {"fixed", "angle"}:
            if angle_unit.lower() in {"degree", "degrees", "deg"}:
                theta = np.deg2rad(angle)
            elif angle_unit.lower() in {"radian", "radians", "rad"}:
                theta = np.asarray(angle, dtype=np.float64)
            else:
                raise ValueError("angle_unit must be 'degrees' or 'radians'.")
        else:
            raise ValueError("rotation_mode must be 'dominant' or 'fixed'.")

        rotated_coefficients = rotate_hermite_coefficients(
            cartesian_coefficients,
            theta,
            max_order=max_order,
            coefficient_region=coefficient_region,
        )

        if "rotated_coefficients" in paths:
            save_coefficients_grid(
                rotated_coefficients,
                paths["rotated_coefficients"],
                "Coeficientes Hermite rotados",
            )
        if "theta" in paths:
            theta_for_plot = _rotation_angle_array(
                theta, next(iter(cartesian_coefficients.values())).shape
            )
            save_theta_image(theta_for_plot, paths["theta"])

    # 3) Rotated -> recovered Cartesian coefficients.
    need_inverse_rotation = use_rotation and (
        use_inverse_rotation or use_inverse_transform
    )
    if need_inverse_rotation:
        recovered_cartesian_coefficients = inverse_rotate_hermite_coefficients(
            rotated_coefficients,
            theta,
            max_order=max_order,
            coefficient_region=coefficient_region,
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

    # 4) Cartesian coefficients -> reconstructed image.
    reconstructed_image = None
    image_metrics = None
    if use_inverse_transform:
        synthesis_coefficients = (
            recovered_cartesian_coefficients
            if use_rotation
            else cartesian_coefficients
        )
        reconstructed_image = inverse_cartesian_hermite_transform(
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

    all_metrics = {}
    if coefficient_metrics is not None:
        all_metrics.update(coefficient_metrics)
    if image_metrics is not None:
        all_metrics.update(image_metrics)
    if "metrics_csv" in paths and all_metrics:
        save_metrics_csv(all_metrics, paths["metrics_csv"])

    # Keep a direct stack for neural-network use, using the MATLAB-compatible
    # order returned by filter_bank['orders'].
    active_coefficients = (
        rotated_coefficients if use_rotation else cartesian_coefficients
    )
    active_stack = _coefficients_to_stack(
        active_coefficients, filter_bank["orders"]
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
        "transformed_image": energy_image(active_coefficients),
        "orders": list(filter_bank["orders"]),
        "filter_bank": filter_bank,
        "output_paths": paths,
    }


if __name__ == "__main__":
    # Example: complete square transform, dominant-orientation rotation,
    # inverse rotation and numerically exact image reconstruction.
    result = hermite_transform_image(
        image="house.tif",
        max_order=3,
        sigma=2.0,
        kernel_size=4,  # exact mode: max_order + 1
        coefficient_region="square",
        exact_reconstruction=True,
        sampling_step=1,
        use_rotation=True,
        use_inverse_rotation=True,
        use_inverse_transform=True,
        rotation_mode="dominant",
        output_paths={
            "cartesian_coefficients": "results/cartesian_coefficients.png",
            "rotated_coefficients": "results/rotated_coefficients.png",
            "recovered_cartesian_coefficients": "results/recovered_cartesian.png",
            "theta": "results/theta.png",
            "reconstruction": "results/reconstruction.png",
            "metrics_csv": "results/metrics.csv",
        },
    )

    print("Orders:", result["orders"])
    print("Coefficient round-trip:", result["coefficient_roundtrip_metrics"])
    print("Reconstruction:", result["reconstruction_metrics"])
