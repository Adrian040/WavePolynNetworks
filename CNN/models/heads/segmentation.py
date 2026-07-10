import torch
import torch.nn as nn


class SegmentationHead(nn.Module):
    # Capa final de segmentación que reduce los canales a la cantidad de clases deseada.
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)