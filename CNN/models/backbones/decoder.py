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

        # Ajusta y normaliza la banda LL aprendida.
        self.reduce_channels = nn.Sequential(
    nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        bias=False
    ),
    nn.BatchNorm2d(out_channels)
)

        # Duplica la resolución usando LL, LH, HL y HH.
        self.idwt = HaarIDWT()

        # Procesa la concatenación de la IDWT y el skip.
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

        # Convierte x en la banda LL del nivel actual.
        x = self.reduce_channels(x)

        # Reconstruye usando LL junto con LH, HL y HH.
        x = self.idwt(x, details)

        # Corrige únicamente diferencias espaciales.
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        # Une la reconstrucción con el skip.
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class Decoder(nn.Module):

    def __init__(
        self,
        features: List[int],
        bilinear: bool = False
    ) -> None:
        super().__init__()

        # bilinear se conserva por compatibilidad,
        # pero la IDWT realiza el aumento de resolución.
        reversed_feats = list(reversed(features))

        self.ups = nn.ModuleList()

        # Canales provenientes del bottleneck.
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
        skip: torch.Tensor,
        details: WaveletDetails
        ) -> torch.Tensor:

        # Convierte x en la banda LL del nivel actual.
        x = self.reduce_channels(x)

        lh, hl, hh = details

        # Diagnóstico de las bandas antes de la IDWT.
        print("\n========== BANDAS ANTES DE IDWT ==========")

        print(
            f"LL -> min: {x.min().item():.4f}, "
            f"max: {x.max().item():.4f}, "
            f"mean: {x.mean().item():.4f}, "
            f"std: {x.std().item():.4f}"
        )

        print(
            f"LH -> min: {lh.min().item():.4f}, "
            f"max: {lh.max().item():.4f}, "
            f"mean: {lh.mean().item():.4f}, "
            f"std: {lh.std().item():.4f}"
        )

        print(
            f"HL -> min: {hl.min().item():.4f}, "
            f"max: {hl.max().item():.4f}, "
            f"mean: {hl.mean().item():.4f}, "
            f"std: {hl.std().item():.4f}"
        )

        print(
            f"HH -> min: {hh.min().item():.4f}, "
            f"max: {hh.max().item():.4f}, "
            f"mean: {hh.mean().item():.4f}, "
            f"std: {hh.std().item():.4f}"
        )

        # Reconstruye usando LL junto con LH, HL y HH.
        x = self.idwt(
            x,
            (lh, hl, hh)
        )

        print(
            f"IDWT -> min: {x.min().item():.4f}, "
            f"max: {x.max().item():.4f}, "
            f"mean: {x.mean().item():.4f}, "
            f"std: {x.std().item():.4f}"
        )

        # Corrige diferencias espaciales.
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        # Une la reconstrucción con el skip.
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)