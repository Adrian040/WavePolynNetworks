"""Visualization helpers corresponding to the MATLAB plotting routines."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ._core import dht2, dhtord, idht2
from ._misc import edht, samplat
from ._steering import gauge


Array = np.ndarray


def _normalize01(X: Array) -> Array:
    arr = np.asarray(X, dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    return np.zeros_like(arr) if lo == hi else (arr - lo) / (hi - lo)


def dhtshow(Y, N: int | Sequence[int], D: int | None = None, *, ax=None, show: bool = False) -> Array:
    """Arrange DHT coefficient maps in the same order mosaic as ``dhtshow.m``."""

    if isinstance(Y, (list, tuple)):
        if np.atleast_1d(N).size > 1:
            base_n, degree = map(int, np.atleast_1d(N)[:2])
        else:
            base_n, degree = 8, int(N)
        tiles = []
        for level, values in enumerate(Y):
            scale_n = base_n if level == 0 else 6
            arr = np.asarray(values)
            if level < len(Y) - 1:
                arr = np.concatenate((np.ones(arr.shape[:2] + (1,) + arr.shape[3:]), arr), axis=2)
            tiles.append(dhtshow(arr, scale_n, degree))
        height = sum(tile.shape[0] for tile in tiles)
        width = sum(tile.shape[1] for tile in tiles)
        channels = tiles[0].shape[2:] if tiles[0].ndim > 2 else ()
        mosaic = np.ones((height, width) + channels, dtype=float)
        row = col = 0
        for tile in tiles:
            mosaic[row : row + tile.shape[0], col : col + tile.shape[1]] = tile
            row += tile.shape[0]
            col += tile.shape[1]
    else:
        arr = np.asarray(Y, dtype=float)
        degree = int(D if D is not None else N)
        if arr.ndim == 4:
            mosaic = np.stack([np.clip(dhtshow(arr[..., k], int(N), degree), 0, 1) for k in range(arr.shape[3])], axis=-1)
        else:
            if arr.ndim == 2:
                arr = arr[..., None]
            rows, cols = arr.shape[:2]
            side = min(int(N), degree) + 1
            mosaic = np.ones((side * rows, side * cols), dtype=float)
            for channel, (horizontal, vertical) in enumerate(dhtord(int(N), degree, 2)):
                if channel >= arr.shape[2]:
                    break
                r = slice(vertical * rows, (vertical + 1) * rows)
                c = slice(horizontal * cols, (horizontal + 1) * cols)
                bounds = np.array([-0.2, 0.2]) * (2 if channel == 0 else 1) + (0.5 if channel == 0 else 0)
                mosaic[r, c] = edht(arr[..., channel], bounds)
    if ax is not None or show:
        import matplotlib.pyplot as plt

        target = ax or plt.subplots()[1]
        target.imshow(mosaic, cmap="gray", vmin=0, vmax=1)
        target.axis("off")
        if show:
            plt.show()
    return mosaic


def matshow(X: Array, k: float = 3, *, ax=None, show: bool = False) -> Array:
    """Normalize every channel by mean and standard deviation for display."""

    arr = np.asarray(X, dtype=float).copy()
    if arr.ndim == 2:
        arr = arr[..., None]
    for channel in range(arr.shape[2]):
        std = np.std(arr[..., channel])
        arr[..., channel] = 0.5 if std == 0 else (arr[..., channel] - np.mean(arr[..., channel])) / (k * std) + 0.5
    result = np.clip(np.squeeze(arr), 0, 1)
    if ax is not None or show:
        import matplotlib.pyplot as plt

        target = ax or plt.subplots()[1]
        target.imshow(result, cmap="gray")
        target.axis("off")
        if show:
            plt.show()
    return result


def graficaMapCoefs(Y: Array, N: int, D: int, *, ax=None, show: bool = False) -> Array:
    """Mosaic in which each coefficient map is independently normalized."""

    arr = np.asarray(Y, dtype=float)
    rows, cols = arr.shape[:2]
    side = min(N, D) + 1
    mosaic = np.ones((side * rows, side * cols), dtype=float)
    for channel, (horizontal, vertical) in enumerate(dhtord(N, D, 2)):
        if channel >= arr.shape[2]:
            break
        mosaic[vertical * rows : (vertical + 1) * rows, horizontal * cols : (horizontal + 1) * cols] = _normalize01(arr[..., channel])
    if ax is not None or show:
        import matplotlib.pyplot as plt

        target = ax or plt.subplots()[1]
        target.imshow(mosaic, cmap="gray")
        target.axis("off")
        if show:
            plt.show()
    return mosaic


grafica = graficaMapCoefs


def angshow(Y, xsiz: Sequence[int], N=None, T=None, shape: str = "full", angle: str = "grad", *, ax=None):
    """Display/return the bidirectional orientation quiver used by ``angshow.m``."""

    if isinstance(Y, (list, tuple)):
        mode = shape if N is None else str(N)
        angle_name = angle if T is None else str(T)
        return [angshow(values if level == len(Y) - 1 else np.concatenate((np.ones(values.shape[:2] + (1,)), values), axis=2), xsiz, 6 * 2 ** (level + 1) - 4, 2 ** (level + 1), mode, angle_name, ax=ax) for level, values in enumerate(Y)]
    arr = np.asarray(Y, dtype=float)
    if N is None or T is None:
        raise ValueError("single-scale angshow requires N and T")
    p = samplat(np.arange(1, int(xsiz[0]) + 1), N, T, shape)
    q = samplat(np.arange(1, int(xsiz[1]) + 1), N, T, shape)
    key = str(angle).lower()
    if key == "grad":
        theta, strength = 2 * np.pi * arr[..., 2] + np.pi / 2, arr[..., 1]
    elif key == "hess":
        theta, strength = 2 * np.pi * arr[..., 4] + np.pi / 2, arr[..., 3] - arr[..., 5]
    else:
        theta = 2 * np.pi * arr[..., -1] + np.pi / 2
        _, strength = gauge(arr, N, 2 * N, int(key), components=True)
    # Preserve the final assignment in the MATLAB source.
    strength = 30 * arr[..., 1]
    qq, pp = np.meshgrid(q, p)
    u, v = strength * np.cos(theta), strength * np.sin(theta)
    if ax is not None:
        first = ax.quiver(qq, pp, u, v, angles="xy", scale_units="xy", scale=1, pivot="middle")
        second = ax.quiver(qq, pp, -u, -v, angles="xy", scale_units="xy", scale=1, pivot="middle")
        return first, second
    return qq, pp, u, v


def clssplot(X: Array, N: int, T: int, tau: Array, theta: Array, style: str = ".r-bog", *, ax=None):
    """Overlay 0-D/1-D/2-D perceptual classes on an image."""

    import matplotlib.pyplot as plt

    target = ax or plt.subplots()[1]
    image = np.asarray(X)
    classes = np.sum(np.asarray(tau), axis=2)
    x = samplat(np.arange(1, image.shape[1] + 1), N, T)
    y = samplat(np.arange(1, image.shape[0] + 1), N, T)
    target.imshow(image, cmap="gray", extent=(1, image.shape[1], image.shape[0], 1))
    handles = []
    for label, marker, color in ((0, ".", "r"), (2, "o", "g")):
        rows, cols = np.where(classes == label)
        handles.append(target.plot(x[cols], y[rows], marker, color=color, markersize=4, linestyle="none")[0])
    rows, cols = np.where(classes == 1)
    if rows.size:
        half = T / 2
        u = half * np.cos(np.pi / 2 + np.asarray(theta)[rows, cols])
        v = half * np.sin(np.pi / 2 + np.asarray(theta)[rows, cols])
        handles.insert(1, target.quiver(x[cols], y[rows], 2 * u, 2 * v, color="b", pivot="middle", angles="xy", scale_units="xy", scale=1))
    target.legend(handles, ["0-D", "1-D", "2-D"][: len(handles)])
    return handles


def guidht(image=None, N: int = 8, D: int = 4, T: int = 2, shape: str = "symm"):
    """Launch a compact Python replacement for the legacy GUIDE-based GUI.

    It exposes the scientifically relevant workflow—load image, transform,
    inspect coefficients, and reconstruct—without depending on MATLAB GUIDE.
    """

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider
    from PIL import Image

    if image is None:
        candidate = Path(__file__).with_name("lena.jpg")
        data = np.asarray(Image.open(candidate).convert("L"), dtype=float) / 255
    elif isinstance(image, (str, Path)):
        data = np.asarray(Image.open(image).convert("L"), dtype=float) / 255
    else:
        data = np.asarray(image, dtype=float)
        if data.max(initial=0) > 1.5:
            data /= 255

    figure, axes = plt.subplots(1, 3, figsize=(13, 5))
    plt.subplots_adjust(bottom=0.22)
    axes[0].imshow(data, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input")
    state = {"N": int(N), "D": int(D), "T": int(T)}

    def refresh(_=None):
        n = int(slider_n.val)
        d = min(int(slider_d.val), 2 * n)
        t = min(max(1, int(slider_t.val)), max(n, 1))
        coeff = dht2(data, n, d, t, shape)
        reconstruction = idht2(coeff, data.shape, n, d, t, shape)
        axes[1].clear()
        axes[1].imshow(dhtshow(coeff, n, d), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"DHT coefficients (N={n}, D={d}, T={t})")
        axes[2].clear()
        axes[2].imshow(reconstruction, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Reconstruction")
        for axis in axes:
            axis.axis("off")
        state.update(N=n, D=d, T=t, coefficients=coeff, reconstruction=reconstruction)
        figure.canvas.draw_idle()

    slider_n = Slider(figure.add_axes((0.15, 0.12, 0.55, 0.03)), "N", 1, 16, valinit=N, valstep=1)
    slider_d = Slider(figure.add_axes((0.15, 0.08, 0.55, 0.03)), "D", 0, 32, valinit=D, valstep=1)
    slider_t = Slider(figure.add_axes((0.15, 0.04, 0.55, 0.03)), "T", 1, 16, valinit=T, valstep=1)
    button = Button(figure.add_axes((0.75, 0.055, 0.16, 0.07)), "Recompute")
    button.on_clicked(refresh)
    refresh()
    figure._dhts_state = state  # keep state and widgets alive for research use
    figure._dhts_widgets = (slider_n, slider_d, slider_t, button)
    return figure


guidhtq = guidht
