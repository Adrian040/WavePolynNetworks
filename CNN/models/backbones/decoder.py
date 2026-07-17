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

        # 1. Reduce canales del bottleneck (1x1 + GroupNorm para estabilidad)
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

        # 2. Reconstrucción Wavelet.
        self.idwt = HaarIDWT()

        # 3. ¡NUEVO! Escala aprendible para controlar las frecuencias altas.
        #    Si el ruido es mucho, el modelo aprenderá a bajar este valor (cerca de 0).
        self.detail_scale = nn.Parameter(
            torch.ones(1, out_channels, 1, 1)
        )

        # 4. ¡NUEVO! Normalización POST-IDWT para igualar escala con el skip.
        #    Esto evita que la IDWT domine la concatenación.
        self.post_norm = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels
        )

        # 5. Procesa la unión entre la reconstrucción y el skip.
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

        # Prepara la banda LL
        x = self.reduce_channels(x)

        # Recupera las bandas de detalle
        lh, hl, hh = details

        # --- APLICAMOS LA ESCALA A LAS FRECUENCIAS ALTAS ---
        # Esto permite que la red "decida" cuánto detalle fino inyectar.
        lh = lh * self.detail_scale
        hl = hl * self.detail_scale
        hh = hh * self.detail_scale

        # Reconstruye el siguiente nivel espacial (IDWT)
        x = self.idwt(x, (lh, hl, hh))

        # --- NORMALIZAMOS LA SALIDA DE LA IDWT ---
        # Ahora x tendrá media ~0 y std ~1, igual que el skip.
        x = self.post_norm(x)

        # --- RESPALDO DE SEGURIDAD PARA DIMENSIONES ---
        # A veces, si la imagen no es divisible por 2^n, la IDWT da 1px de menos.
        # Lo arreglamos con interpolación SIN perder la magia de las frecuencias.
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        # Une la salida reconstruida con la conexión skip.
        x = torch.cat([skip, x], dim=1)

        # Refina la fusión
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