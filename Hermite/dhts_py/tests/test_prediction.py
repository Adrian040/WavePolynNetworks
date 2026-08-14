"""Python version of the multiscale prediction experiment pred_test.m."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dhts_py.mdht2 import mdht2
from dhts_py.pdht import pdht


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def run_pred_test(image=None, N0: int = 8, D: int = 12, M: int = 4, level: int = 2):
    if image is None:
        image = PACKAGE_DIR / "lena.jpg"
    if isinstance(image, (str, Path)):
        X = np.asarray(Image.open(image).convert("L"), dtype=float) / 255
    else:
        X = np.asarray(image, dtype=float)
    original = mdht2(X, [N0, D], M, "symm")
    residues = mdht2(X, [N0, D], M, "symm", "p")
    residues[level] = np.zeros_like(residues[level])
    predicted = pdht(residues, D, "inv", "symm")
    metrics = {}
    for channel in range(min(5, original[level].shape[2], predicted[level].shape[2])):
        truth, estimate = original[level][..., channel], predicted[level][..., channel]
        noise_std = np.std(truth - estimate)
        metrics[channel] = float("inf") if noise_std == 0 else float(20 * np.log10(np.std(truth) / noise_std))
    return original, predicted, metrics


if __name__ == "__main__":
    _, _, snr = run_pred_test()
    print("prediction SNR by channel:", snr)
