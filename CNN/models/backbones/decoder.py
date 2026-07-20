import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from ..blocks.conv import DoubleConv
from Wavelet.idwt import HaarIDWT


WaveletDetails = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor
]


class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ) -> None:
        super().__init__()

        # Reduce canales
        self.reduce_channels = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=out_channels
            )
        )

        # Reconstrucción Wavelet
        self.idwt = HaarIDWT()

        # Escala de detalles
        self.detail_scale = nn.Parameter(
            torch.ones(1, out_channels, 1, 1)
        )

        # Normalización
        self.post_norm = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels
        )

        # Fusión con skip
        self.conv = DoubleConv(
            out_channels * 2,
            out_channels
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        details: WaveletDetails
    ) -> torch.Tensor:

        # Banda LL
        x = self.reduce_channels(x)

        # Bandas de detalle
        lh, hl, hh = details

        # Escala detalles
        lh = lh * self.detail_scale
        hl = hl * self.detail_scale
        hh = hh * self.detail_scale

        # Reconstrucción
        x = self.idwt(x, (lh, hl, hh))

        # Normaliza salida
        x = self.post_norm(x)

        # Ajusta tamaño 
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        # Concatena skip
        x = torch.cat([skip, x], dim=1)

        # Refina salida
        return self.conv(x)

class Decoder(nn.Module):

    def __init__(
        self,
        features: List[int],
        bilinear: bool = False
    ) -> None:
        super().__init__()

        reversed_feats = list(reversed(features))
        self.ups = nn.ModuleList()

        in_channels = reversed_feats[0] * 2  # Bottleneck

        for out_channels in reversed_feats:
            self.ups.append(
                UpBlock(
                    in_channels=in_channels,
                    out_channels=out_channels
                )
            )
            in_channels = out_channels

    def forward(
        self,
        x: torch.Tensor,
        skips: List[torch.Tensor],
        wavelet_details: List[WaveletDetails]
    ) -> torch.Tensor:

        skips = list(reversed(skips))
        wavelet_details = list(reversed(wavelet_details))

        for up, skip, details in zip(
            self.ups,
            skips,
            wavelet_details
        ):
            x = up(x, skip, details)

        return x