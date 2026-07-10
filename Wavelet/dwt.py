import torch
import torch.nn as nn
import torch.nn.functional as F

class HaarDWT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Filtro LL: conserva la información general.
        ll = torch.tensor([
            [1.0, 1.0],
            [1.0, 1.0]
        ]) / 2.0
        # Filtro LH: detecta cambios verticales.
        lh = torch.tensor([
            [-1.0, -1.0],
            [ 1.0,  1.0]
        ]) / 2.0
         # Filtro HL: detecta cambios horizontales.
        hl = torch.tensor([
            [-1.0, 1.0],
            [-1.0, 1.0]
        ]) / 2.0
        # Filtro HH: detecta cambios diagonales.
        hh = torch.tensor([
            [ 1.0, -1.0],
            [-1.0,  1.0]
        ]) / 2.0
        # Stack el filtro en un tensor de 4 dimensiones para usarlo en la convolución.
        filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1)
        # Guarda los filtros sin convertirlos en parámetros entrenables.
        self.register_buffer("filters", filters)

    def forward(self, x: torch.Tensor):
        # Obtiene las dimensiones de la entrada.
        batch_size, channels, height, width = x.shape

        # Repite los cuatro filtros para cada canal.
        filters = self.filters.repeat(channels, 1, 1, 1)

        # Aplica los filtros Haar y reduce el tamaño a la mitad.
        output = F.conv2d(
            x,
            filters,
            stride=2,
            groups=channels
        )
        # Organiza la salida por canal y por tipo de banda.
        output = output.view(
            batch_size,
            channels,
            4,
            height // 2,
            width // 2
        )
        # Separa las cuatro bandas Wavelet.
        ll = output[:, :, 0]
        lh = output[:, :, 1]
        hl = output[:, :, 2]
        hh = output[:, :, 3]

        # Devuelve la información general y los detalles.
        return ll, (lh, hl, hh)