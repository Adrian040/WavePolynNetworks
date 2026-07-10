import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarIDWT(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Filtro LL: reconstruye la información general.
        ll = torch.tensor([
            [1.0, 1.0],
            [1.0, 1.0]
        ]) / 2.0

        # Filtro LH: reconstruye los cambios verticales.
        lh = torch.tensor([
            [-1.0, -1.0],
            [ 1.0,  1.0]
        ]) / 2.0

        # Filtro HL: reconstruye los cambios horizontales.
        hl = torch.tensor([
            [-1.0, 1.0],
            [-1.0, 1.0]
        ]) / 2.0

        # Filtro HH: reconstruye los cambios diagonales.
        hh = torch.tensor([
            [ 1.0, -1.0],
            [-1.0,  1.0]
        ]) / 2.0

        # Junta los filtros y agrega la dimensión del canal.
        filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1)

        # Guarda los filtros sin hacerlos entrenables.
        self.register_buffer("filters", filters)

    def forward(
        self,
        ll: torch.Tensor,
        details: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor
        ]
    ) -> torch.Tensor:

        # Obtiene las bandas de detalle.
        lh, hl, hh = details

        # Verifica que todas tengan el mismo tamaño.
        if not (
            ll.shape == lh.shape == hl.shape == hh.shape
        ):
            raise ValueError(
                "LL, LH, HL y HH deben tener las mismas dimensiones."
            )

        # Obtiene las dimensiones de una banda.
        batch_size, channels, height, width = ll.shape

        # Junta las cuatro bandas.
        output = torch.stack(
            [ll, lh, hl, hh],
            dim=2
        )

        # Cambia de [B, C, 4, H, W] a [B, C*4, H, W].
        output = output.view(
            batch_size,
            channels * 4,
            height,
            width
        )

        # Repite los filtros para cada canal.
        filters = self.filters.repeat(
            channels,
            1,
            1,
            1
        )

        # Reconstruye y duplica el tamaño espacial.
        reconstructed = F.conv_transpose2d(
            output,
            filters,
            stride=2,
            groups=channels
        )

        return reconstructed
