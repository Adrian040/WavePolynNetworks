"""Python version of the numerical reconstruction test in dht2_test.m."""

from pathlib import Path

import numpy as np
from PIL import Image

from .dht2 import dht2
from .idht2 import idht2


def run_dht2_test(image=None, N: int = 8, D: int | None = None, T: int = 2):
    if image is None:
        image = Path(__file__).with_name("lena.jpg")
    if isinstance(image, (str, Path)):
        X = np.asarray(Image.open(image).convert("L"), dtype=float) / 255
    else:
        X = np.asarray(image, dtype=float)
    D = 2 * N if D is None else int(D)
    Y = dht2(X, N, D, T)
    X2 = idht2(Y, X.shape, N, D, T)
    error = X - X2
    return {"coefficients": Y, "reconstruction": X2, "max_abs_error": float(np.max(np.abs(error))), "mse": float(np.mean(error**2))}


if __name__ == "__main__":
    result = run_dht2_test()
    print(f"max_abs_error={result['max_abs_error']:.3e}, mse={result['mse']:.3e}")

