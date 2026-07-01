import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss multiclase. Espera logits (N, C, H, W) y targets (N, H, W) con
    índices de clase (igual que CrossEntropyLoss).
    """

    def __init__(self, num_classes: int, ignore_background: bool = False, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)                       # (N, C, H, W)
        targets_onehot = F.one_hot(targets, self.num_classes)  # (N, H, W, C)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # (N, C, H, W)

        start_class = 1 if self.ignore_background else 0

        dims = (0, 2, 3)  # suma sobre batch y espacio, deja las clases
        intersection = (probs * targets_onehot).sum(dims)
        union = probs.sum(dims) + targets_onehot.sum(dims)

        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_per_class = dice_per_class[start_class:]  # excluye fondo si aplica

        return 1 - dice_per_class.mean()