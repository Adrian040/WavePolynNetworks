import torch
import torch.nn as nn

from .hermite_filters import HermiteGaussianBasis


class HermiteBottleneck(nn.Module):

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 256,
        kernel_size: int = 5,
        sigma: float = 1.0,
    ):
        super().__init__()

        # Número de canales de entrada y salida
        self.channels = channels

        # Reduce los canales antes de aplicar Hermite
        self.reduce = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Aplica los seis filtros Hermite-Gauss
        self.hermite = HermiteGaussianBasis(
            channels=hidden_channels,
            kernel_size=kernel_size,
            sigma=sigma,
        )

        # Combina las respuestas y recupera los canales originales
        self.combine = nn.Sequential(
            nn.Conv2d(
                hidden_channels * HermiteGaussianBasis.NUM_FILTERS,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

        # Activación final
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        # Guarda la entrada para la conexión residual
        residual = x

        # Reduce el número de canales
        x = self.reduce(x)

        # Extrae características con Hermite
        x = self.hermite(x)

        # Combina las respuestas de los filtros
        x = self.combine(x)

        # Suma la entrada original
        x = x + residual

        # Devuelve la salida activada
        return self.relu(x)