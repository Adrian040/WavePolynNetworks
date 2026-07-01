import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class LungDataset(Dataset):
    """
    Dataset para cortes de CT (float32, valores en Hounsfield Units) con
    máscaras multiclase: 0=fondo, 1=ground-glass opacity, 2=consolidación.
    """

    def __init__(self, images_dir, masks_dir, img_size=256,
                 hu_min=-1000, hu_max=400):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir

        # Solo nombres que existen en ambas carpetas (evita pares rotos)
        img_names  = set(os.listdir(images_dir))
        mask_names = set(os.listdir(masks_dir))
        self.filenames = sorted(img_names & mask_names)

        if len(self.filenames) == 0:
            raise RuntimeError("No se encontraron pares imagen-máscara con el mismo nombre.")

        self.img_size = img_size
        self.hu_min = hu_min   # límite inferior de la ventana de CT
        self.hu_max = hu_max   # límite superior de la ventana de CT

        # NEAREST para no crear valores de clase intermedios al hacer resize de la máscara
        self.resize_img  = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)
        self.resize_mask = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.filenames)

    def _load_image(self, path):
        arr = np.array(Image.open(path)).astype(np.float32)  # HU crudos

        # Windowing: recorta al rango clínico relevante y normaliza a [0,1]
        arr = np.clip(arr, self.hu_min, self.hu_max)
        arr = (arr - self.hu_min) / (self.hu_max - self.hu_min)

        return arr

    def _load_mask(self, path):
        arr = np.array(Image.open(path)).astype(np.int64)  # clases: 0,1,2
        return arr

    def __getitem__(self, idx):
        name = self.filenames[idx]

        img_arr  = self._load_image(os.path.join(self.images_dir, name))
        mask_arr = self._load_mask(os.path.join(self.masks_dir, name))

        img  = torch.from_numpy(img_arr).unsqueeze(0)   # (1, H, W) float
        mask = torch.from_numpy(mask_arr).unsqueeze(0)  # (1, H, W) long (temporal, para resize)

        img  = self.resize_img(img)
        mask = self.resize_mask(mask)

        mask = mask.squeeze(0).long()  # (H, W), sin canal — así lo pide CrossEntropyLoss

        return img, mask