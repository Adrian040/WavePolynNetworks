import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
import tifffile as tiff  

def _extract_id(filename: str) -> str:
    """Extrae el número identificador del archivo, ej. 'tr_im0000.tif' -> '0000'."""
    match = re.search(r"(\d+)", filename)
    if match is None:
        raise ValueError(f"No se encontró número identificador en: {filename}")
    return match.group(1)

class LungDataset(Dataset):
    def __init__(self, images_dir, masks_dir, pairs, transform=None, hu_min=-1000, hu_max=400):
        """
        pairs: Lista de tuplas (nombre_imagen, nombre_mascara) ya divididas.
        transform: Pipeline de Albumentations.
        """
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.pairs = pairs
        self.transform = transform
        self.hu_min = hu_min
        self.hu_max = hu_max

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path):
        # MODIFICACIÓN: Usar tifffile en lugar de PIL para preservar los datos crudos del CT
        arr = tiff.imread(path).astype(np.float32)
        arr = np.clip(arr, self.hu_min, self.hu_max)
        arr = (arr - self.hu_min) / (self.hu_max - self.hu_min)
        return arr # Retorna array 2D en rango [0, 1]

    def _load_mask(self, path):
        # MODIFICACIÓN: Usar tifffile también para la máscara
        arr = tiff.imread(path).astype(np.int64)
        return arr # Retorna array 2D

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]

        img_arr  = self._load_image(os.path.join(self.images_dir, img_name))
        mask_arr = self._load_mask(os.path.join(self.masks_dir, mask_name))

        # Aplicar transformaciones simultáneas (Data Augmentation)
        if self.transform is not None:
            augmented = self.transform(image=img_arr, mask=mask_arr)
            img_arr = augmented['image']
            mask_arr = augmented['mask']

        # Convertir a tensores
        # La CNN espera (Canales, Alto, Ancho), CT es 1 canal, así que hacemos unsqueeze(0)
        img  = torch.from_numpy(img_arr).unsqueeze(0).float()
        
        # La máscara para CrossEntropy debe ser (Alto, Ancho) en formato Long (sin dimensión de canal)
        mask = torch.from_numpy(mask_arr).long()

        return img, mask