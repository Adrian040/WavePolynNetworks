import torch
import torch.nn as nn


class SegmentationHead(nn.Module):
    """
    Cabeza de salida para segmentación.

    Conv1x1 → reduce los feature maps al número de clases.
    La activación (sigmoid/softmax) se aplica FUERA del modelo,
    en la función de pérdida (BCEWithLogitsLoss / CrossEntropyLoss).

    Args:
        in_channels  : canales que llegan del decoder (= features[0])
        out_channels : número de clases (1=binario, N=multiclase)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)