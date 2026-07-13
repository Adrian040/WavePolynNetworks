import torch
import torch.nn as nn
from typing import List
from ..blocks.conv import DoubleConv
from Wavelet.dwt import HaarDWT
class Encoder(nn.Module):
    def __init__(self, in_channels: int, features: List[int]) -> None:
        super().__init__()

        self.downs = nn.ModuleList() # Lista dinámica de bloques del encoder
        self.dwt = HaarDWT()  # Reduce la resolución mediante Wavelet Haar
    

        ch = in_channels 
        for feat in features:
            self.downs.append(DoubleConv(ch, feat)) # Bloque conv: extrae y refina características
            ch = feat

        # Bottleneck: doble de filtros del último nivel
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

    def forward(self, x: torch.Tensor):
        skips = []
        wavelet_details = []

        for down in self.downs:
            x = down(x)
            skips.append(x)

            # Aplica la DWT y guarda las bandas de detalle.
            x, details = self.dwt(x)
            wavelet_details.append(details)

        x = self.bottleneck(x)

        return x, skips, wavelet_details