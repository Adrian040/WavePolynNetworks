import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from ..blocks.conv import DoubleConv
from Wavelet.idwt import HaarIDWT


class UpBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        # Ajusta los canales de x para que coincidan con LH, HL y HH.
        self.reduce_channels = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

        # Reconstruye el tamaño usando las cuatro bandas Wavelet.
        self.idwt = HaarIDWT()

        # Después de concatenar: skip + reconstrucción.
        self.conv = DoubleConv(
            out_channels * 2,
            out_channels
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        details: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor
        ]
    ) -> torch.Tensor:

        # Reduce los canales para formar la banda LL.
        x = self.reduce_channels(x)

        # Reconstruye: LL + LH + HL + HH.
        x = self.idwt(x, details)

        # Ajusta el tamaño espacial si fuera necesario.
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        # Une la reconstrucción con el skip de la U-Net.
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class Decoder(nn.Module):

    def __init__(
        self,
        features: List[int],
        bilinear: bool = False
    ) -> None:
        super().__init__()

        # Ejemplo: [64, 128, 256, 512]
        reversed_feats = list(reversed(features))

        self.ups = nn.ModuleList()

        # Salida del bottleneck: 512 × 2 = 1024 canales.
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
        wavelet_details: List[
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor
            ]
        ]
    ) -> torch.Tensor:

        # El decoder empieza desde el nivel más profundo.
        skips = list(reversed(skips))
        wavelet_details = list(reversed(wavelet_details))

        for up, skip, details in zip(
            self.ups,
            skips,
            wavelet_details
        ):
            x = up(x, skip, details)

        return x