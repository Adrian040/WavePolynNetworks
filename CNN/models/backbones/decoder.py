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


# ---------------------------------------------------------
# 1. Bloque para refinar LH, HL y HH sin borrar la subbanda
# ---------------------------------------------------------
class DetailRefine(nn.Module):

    def __init__(
        self,
        channels: int
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=channels
            ),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        )

        # Controla cuánto modifica la subbanda original.
        self.alpha = nn.Parameter(
            torch.tensor(0.1)
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        # Subbanda original + corrección aprendida.
        return x + self.alpha * self.block(x)


# ---------------------------------------------------------
# 2. Bloque de subida del decoder
# ---------------------------------------------------------
class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ) -> None:
        super().__init__()

        # Procesa la LL que viene del bottleneck
        # o del UpBlock anterior.
        self.process_ll = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=out_channels
            ),
            nn.GELU()
        )

        # Procesa las subbandas del encoder.
        self.process_lh = DetailRefine(
            out_channels
        )

        self.process_hl = DetailRefine(
            out_channels
        )

        self.process_hh = DetailRefine(
            out_channels
        )

        # Reconstrucción Wavelet.
        self.idwt = HaarIDWT()

        # Normaliza la salida reconstruida.
        self.align_norm = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels
        )

        # Fusiona la IDWT con el skip.
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

        # 1. Procesar LL.
        ll = self.process_ll(x)

        # 2. Recuperar detalles del encoder.
        lh, hl, hh = details

        # 3. Refinar detalles conservando los originales.
        lh = self.process_lh(lh)
        hl = self.process_hl(hl)
        hh = self.process_hh(hh)

        # 4. Reconstrucción Wavelet.
        x = self.idwt(
            ll,
            (lh, hl, hh)
        )

        # 5. Ajustar tamaño si fuera necesario.
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        # 6. Normalizar reconstrucción.
        x = self.align_norm(x)

        # 7. Concatenar con skip.
        x = torch.cat(
            [skip, x],
            dim=1
        )

        # 8. Procesar características fusionadas.
        return self.conv(x)


# ---------------------------------------------------------
# 3. Decoder completo
# ---------------------------------------------------------
class Decoder(nn.Module):

    def __init__(
        self,
        features: List[int],
        bilinear: bool = False
    ) -> None:
        super().__init__()

        reversed_feats = list(
            reversed(features)
        )

        self.ups = nn.ModuleList()

        # El bottleneck tiene el doble de canales.
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

        # Iniciar desde el nivel más profundo.
        skips = list(
            reversed(skips)
        )

        wavelet_details = list(
            reversed(wavelet_details)
        )

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