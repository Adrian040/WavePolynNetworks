"""API pública de la transformada Hermite--Gaussiana 2-D."""

from .analysis import cartesian_transform
from .filters import (
    build_filter_bank,
    choose_support,
    hermite_filter_1d,
    hermite_orders,
)
from .image_io import read_image
from .metrics import (
    coefficient_energy,
    coefficient_roundtrip_metrics,
    coefficients_to_stack,
    reconstruction_metrics,
)
from .steering import (
    dominant_theta,
    inverse_rotate_coefficients,
    rotate_block,
    rotate_coefficients,
)
from .synthesis import synthesize
from .workflow import hermite_transform_image

__all__ = [
    "build_filter_bank",
    "cartesian_transform",
    "choose_support",
    "coefficient_energy",
    "coefficient_roundtrip_metrics",
    "coefficients_to_stack",
    "dominant_theta",
    "hermite_filter_1d",
    "hermite_orders",
    "hermite_transform_image",
    "inverse_rotate_coefficients",
    "read_image",
    "reconstruction_metrics",
    "rotate_block",
    "rotate_coefficients",
    "synthesize",
]
