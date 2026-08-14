"""Python counterpart of the legacy GUIDE application ``guidht.m``."""

from pathlib import Path

import numpy as np

from .dht2 import dht2
from .dhtshow import dhtshow
from .idht2 import idht2


def guidht(image=None, N: int = 8, D: int = 4, T: int = 2, shape: str = "symm"):
    """Launch a compact Matplotlib transform/reconstruction explorer."""

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
        coefficients = dht2(data, n, d, t, shape)
        reconstruction = idht2(coefficients, data.shape, n, d, t, shape)
        axes[1].clear()
        axes[1].imshow(dhtshow(coefficients, n, d), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"DHT coefficients (N={n}, D={d}, T={t})")
        axes[2].clear()
        axes[2].imshow(reconstruction, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Reconstruction")
        for axis in axes:
            axis.axis("off")
        state.update(N=n, D=d, T=t, coefficients=coefficients, reconstruction=reconstruction)
        figure.canvas.draw_idle()

    slider_n = Slider(figure.add_axes((0.15, 0.12, 0.55, 0.03)), "N", 1, 16, valinit=N, valstep=1)
    slider_d = Slider(figure.add_axes((0.15, 0.08, 0.55, 0.03)), "D", 0, 32, valinit=D, valstep=1)
    slider_t = Slider(figure.add_axes((0.15, 0.04, 0.55, 0.03)), "T", 1, 16, valinit=T, valstep=1)
    button = Button(figure.add_axes((0.75, 0.055, 0.16, 0.07)), "Recompute")
    button.on_clicked(refresh)
    refresh()
    figure._dhts_state = state
    figure._dhts_widgets = (slider_n, slider_d, slider_t, button)
    return figure


guidhtq = guidht

__all__ = ["guidht", "guidhtq"]
