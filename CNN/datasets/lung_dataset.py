import os
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def _extract_id(filename: str) -> str:
    """Extrae el número identificador del archivo, ej. 'tr_im0000.tif' -> '0000'."""
    match = re.search(r"(\d+)", filename)
    if match is None:
        raise ValueError(f"No se encontró número identificador en: {filename}")
    return match.group(1)


class LungDataset(Dataset):
    """
    Dataset para cortes de CT (float32, valores en Hounsfield Units) con
    máscaras multiclase: 0=fondo, 1=ground-glass opacity, 2=consolidación.

    Empareja imagen <-> máscara por el número dentro del nombre de archivo
    (ej. 'tr_im0000.tif' con 'tr_mask0000.tif'), no por nombre exacto.
    """

    def __init__(self, images_dir, masks_dir, img_size=256,
                 hu_min=-1000, hu_max=400):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir

        img_files  = os.listdir(images_dir)
        mask_files = os.listdir(masks_dir)

        img_by_id  = {_extract_id(f): f for f in img_files}
        mask_by_id = {_extract_id(f): f for f in mask_files}

        common_ids = sorted(set(img_by_id) & set(mask_by_id))

        if len(common_ids) == 0:
            raise RuntimeError("No se encontraron pares imagen-máscara con el mismo número identificador.")

        # Guarda pares (nombre_imagen, nombre_mascara) ya emparejados
        self.pairs = [(img_by_id[i], mask_by_id[i]) for i in common_ids]

        self.img_size = img_size
        self.hu_min = hu_min
        self.hu_max = hu_max

        self.resize_img  = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)
        self.resize_mask = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path):
        arr = np.array(Image.open(path)).astype(np.float32)
        arr = np.clip(arr, self.hu_min, self.hu_max)
        arr = (arr - self.hu_min) / (self.hu_max - self.hu_min)
        return arr

    def _load_mask(self, path):
        arr = np.array(Image.open(path)).astype(np.int64)
        return arr

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]

        img_arr  = self._load_image(os.path.join(self.images_dir, img_name))
        mask_arr = self._load_mask(os.path.join(self.masks_dir, mask_name))

        img  = torch.from_numpy(img_arr).unsqueeze(0)
        mask = torch.from_numpy(mask_arr).unsqueeze(0)

        img  = self.resize_img(img)
        mask = self.resize_mask(mask)

        mask = mask.squeeze(0).long()

        return img, mask