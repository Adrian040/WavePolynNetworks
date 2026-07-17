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
    """
    Bloque de decoder con reconstrucción Wavelet (IDWT).
    
    IMPORTANTE: Se usa 1x1 Conv + GroupNorm para la banda LL,
    ya que es mucho más estable numéricamente para la IDWT
    que usar DoubleConv con BatchNorm (evita artefactos de anillo).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ) -> None:
        super().__init__()

        # Reduce los canales del bottleneck a la banda LL.
        # Se usa GroupNorm para no depender del batch size.
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

        # Reconstrucción Wavelet (operación fija, no entrenable).
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

        # Reconstruye el siguiente nivel espacial (duplica H y W).
        x = self.idwt(
            x,
            (lh, hl, hh)
        )

        # Verificación estricta de tamaños.
        # Si esto falla, la DWT/IDWT tiene un problema de padding o paridad.
        # No uses interpolación aquí porque romperías las frecuencias altas.
        assert x.shape[2:] == skip.shape[2:], \
            f"Shape mismatch en UpBlock: x={x.shape}, skip={skip.shape}"

        # Une la salida reconstruida con la conexión skip.
        x = torch.cat(
            [skip, x],
            dim=1
        )

        return self.conv(x)


class Decoder(nn.Module):
    """
    Decoder completo de la Wavelet U-Net.
    """

    def __init__(
        self,
        features: List[int],
        bilinear: bool = False
    ) -> None:
        super().__init__()

        # Invertimos la lista de características para el decoder.
        # Ejemplo: [64, 128, 256, 512] -> [512, 256, 128, 64]
        reversed_feats = list(
            reversed(features)
        )

        self.ups = nn.ModuleList()

        # El bottleneck tiene el doble de canales que el último nivel del encoder.
        # Ejemplo: features[-1] = 512 -> in_channels = 1024
        in_channels = reversed_feats[0] * 2

        for out_channels in reversed_feats:

            self.ups.append(
                UpBlock(
                    in_channels=in_channels,
                    out_channels=out_channels
                )
            )

            # La salida de este bloque será la entrada del siguiente.
            in_channels = out_channels

    def forward(
        self,
        x: torch.Tensor,
        skips: List[torch.Tensor],
        wavelet_details: List[WaveletDetails]
    ) -> torch.Tensor:

        # El decoder va desde el nivel más profundo hasta el más superficial.
        skips = list(
            reversed(skips)
        )

        wavelet_details = list(
            reversed(wavelet_details)
        )

        # Cada UpBlock recibe: x, skip y detalles Wavelet.
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