import torch.nn as nn
class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers = []
        for i in range(2):  # dos convoluciones
            in_ch = in_channels if i == 0 else out_channels
            layers.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1, bias=not batch_norm)
            )
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)  # *layers desempaqueta la lista

    def forward(self, x):
        return self.block(x)