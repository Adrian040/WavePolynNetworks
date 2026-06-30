import torch
import torch.nn as nn
from typing import List

from ..blocks.conv import DoubleConv


class UpBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool) -> None:
        super().__init__()

        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # ConvTranspose2d  aprende a interpolar y refinar características
            self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Ajuste de padding si el tamaño no encaja exactamente (inputs no multiplos de 2^n)
        if x.shape != skip.shape:
            x = nn.functional.pad(x, [0, skip.shape[3] - x.shape[3],
                                       0, skip.shape[2] - x.shape[2]])

        x = torch.cat([skip, x], dim=1)   # concatenar por canales 
        return self.conv(x)


class Decoder(nn.Module):
  
    def __init__(self, features: List[int], bilinear: bool = False) -> None:
        super().__init__()

        # features viene de mayor a menor: [512, 256, 128, 64]
        reversed_feats = list(reversed(features))

        self.ups = nn.ModuleList()
        in_ch = reversed_feats[0] * 2   # canales del bottleneck

        for feat in reversed_feats:
            self.ups.append(UpBlock(in_ch, feat, bilinear))
            in_ch = feat

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        skips = list(reversed(skips))

        for up, skip in zip(self.ups, skips):
            x = up(x, skip)

        return x