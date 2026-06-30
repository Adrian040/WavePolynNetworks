import torch
import torch.nn as nn
from .config import UNetConfig
from .backbones.encoder import Encoder
from .backbones.decoder import Decoder
from .heads.segmentation import SegmentationHead


class UNet(nn.Module):
    def __init__(self, config: UNetConfig = None) -> None:
        super().__init__()
        self.config = config or UNetConfig()

        self.encoder = Encoder(
            in_channels=self.config.in_channels,   # Canales de entrada.
            features=self.config.features,         # Filtros por cada nivel.
        )

        self.decoder = Decoder(
            features=self.config.features,         # Filtros por cada nivel (inverso).
            bilinear=self.config.bilinear,         # Tipo de upsampling.
        )

        self.head = SegmentationHead(
            in_channels=self.config.features[0],  # Canales finales del decoder.
            out_channels=self.config.out_channels,# Canales de salida.
        )

        self._init_weights()  # Inicializa los pesos.
   # Flujo principal del modelo U-Net
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skips   = self.encoder(x) 
        x          = self.decoder(x, skips)
        return self.head(x)

   # Inicializa los pesos del modelo.
    def _init_weights(self) -> None:
        for m in self.modules(): 
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): 
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")  
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_config(self, path: str) -> None:
        self.config.save(path)

    @classmethod
    def from_config(cls, path: str) -> "UNet":
        return cls(UNetConfig.load(path))