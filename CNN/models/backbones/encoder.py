import torch
import torch.nn as nn
from typing import List
from ..blocks.conv import DoubleConv
class Encoder(nn.Module):
    def __init__(self, in_channels: int, features: List[int]) -> None:
        super().__init__()

        self.downs = nn.ModuleList() # Lista dinámica de bloques del encoder
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2) # Reduce la resolución a la mitad

        ch = in_channels 
        for feat in features:
            self.downs.append(DoubleConv(ch, feat)) # Bloque conv: extrae y refina características
            ch = feat

        # Bottleneck: doble de filtros del último nivel
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

    def forward(self, x: torch.Tensor):
        skips = []
        for down in self.downs:
            x = down(x) 
            skips.append(x)   
            x = self.pool(x)

        x = self.bottleneck(x)
        return x, skips