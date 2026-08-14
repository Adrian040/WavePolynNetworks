"""Direct translation of ``clssplot.m``."""

import numpy as np

from .samplat import samplat


def clssplot(X, N: int, T: int, tau, theta, style: str = ".r-bog", *, ax=None):
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


__all__ = ["clssplot"]
