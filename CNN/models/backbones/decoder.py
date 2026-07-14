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

        # Ajusta los canales de LL y normaliza sin depender del batch.
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

        # Reconstrucción Wavelet.
        self.idwt = HaarIDWT()

        # Procesa la unión entre la reconstrucción y el skip.
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

        # Convierte x en la nueva banda LL.
        x = self.reduce_channels(x)

        # Recupera las bandas del encoder.
        lh, hl, hh = details

        # Reconstruye el siguiente nivel.
        x = self.idwt(
            x,
            (lh, hl, hh)
        )

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        # Une la salida reconstruida con la conexión skip.
        x = torch.cat(
            [skip, x],
            dim=1
        )

        return self.conv(x)


class Decoder(nn.Module):

    def __init__(
        self,
        features: List[int],
        bilinear: bool = False
    ) -> None:
        super().__init__()

        # Ejemplo:
        # [64, 128, 256, 512]
        # se convierte en:
        # [512, 256, 128, 64]
        reversed_feats = list(
            reversed(features)
        )

        self.ups = nn.ModuleList()

        # El bottleneck tiene el doble de canales
        # que el último nivel del encoder.
        in_channels = reversed_feats[0] * 2

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

        # Empieza desde el nivel más profundo.
        skips = list(
            reversed(skips)
        )

        wavelet_details = list(
            reversed(wavelet_details)
        )

        # Cada UpBlock recibe:
        # x, skip y detalles Wavelet.
        for up, skip, details in zip(
            self.ups,
            skips,
            wavelet_details
        ):
            x = up(
                x,
                skip,
                details
            )

        return x