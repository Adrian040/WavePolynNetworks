import torch
import torch.nn as nn
import torch.nn.functional as F


class HermiteGaussianBasis(nn.Module):

    NUM_FILTERS = 6

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        sigma: float = 1.0,
    ):
        super().__init__()

        # Número de canales que recibirá el bloque
        self.channels = channels

        # Mantiene el mismo tamaño espacial
        self.padding = kernel_size // 2

        # Guarda los filtros fijos dentro del modelo
        self.register_buffer(
            "filters",
            self._build_filters(kernel_size, sigma),
        )

    @staticmethod
    def _normalize(kernel):

        # Normaliza cada filtro para evitar valores muy grandes
        return kernel / torch.linalg.vector_norm(kernel).clamp_min(1e-8)

    @classmethod
    def _build_filters(cls, kernel_size, sigma):

        # Calcula el radio del kernel
        radius = kernel_size // 2

        # Crea las coordenadas del filtro
        coords = torch.arange(
            -radius,
            radius + 1,
            dtype=torch.float32,
        )

        # Genera la cuadrícula 2D
        y, x = torch.meshgrid(
            coords,
            coords,
            indexing="ij",
        )

        # Ajusta las coordenadas con sigma
        xn = x / sigma
        yn = y / sigma

        # Calcula la función gaussiana
        gaussian = torch.exp(
            -0.5 * (xn ** 2 + yn ** 2)
        )

        # Hermite de orden 0
        h0x = torch.ones_like(xn)
        h0y = torch.ones_like(yn)

        # Hermite de orden 1
        h1x = xn
        h1y = yn

        # Hermite de orden 2
        h2x = xn ** 2 - 1
        h2y = yn ** 2 - 1

        # Crea los seis filtros Hermite-Gauss
        basis = [
            h0x * h0y * gaussian,
            h1x * h0y * gaussian,
            h0x * h1y * gaussian,
            h2x * h0y * gaussian,
            h1x * h1y * gaussian,
            h0x * h2y * gaussian,
        ]

        # Normaliza cada filtro
        basis = [
            cls._normalize(k)
            for k in basis
        ]

        # Forma final: [6, 1, kernel, kernel]
        return torch.stack(
            basis,
            dim=0,
        ).unsqueeze(1)

    def forward(self, x):

        # Repite los seis filtros para cada canal
        weights = self.filters.repeat(
            self.channels,
            1,
            1,
            1,
        )

        # Aplica los filtros de forma independiente por canal
        return F.conv2d(
            x,
            weights,
            stride=1,
            padding=self.padding,
            groups=self.channels,
        )