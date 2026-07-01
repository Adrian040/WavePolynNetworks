from dataclasses import dataclass, field
from typing import List
import json


@dataclass
class UNetConfig:
    in_channels: int = 1                  # canales de entrada (1=grayscale, 3=RGB)
    out_channels: int = 3                # canales de salida (1=binario, N=multiclase)
    features: List[int] = field(          # filtros por nivel del encoder
        default_factory=lambda: [64, 128, 256, 512]
    )
    bilinear: bool = False                # False=ConvTranspose2d (paper original)
    dropout: float = 0.0                  # sin dropout en la clásica
    def to_dict(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "features": self.features,
            "bilinear": self.bilinear,
            "dropout": self.dropout,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "UNetConfig":
        return cls(**d)

    @classmethod
    def load(cls, path: str) -> "UNetConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))