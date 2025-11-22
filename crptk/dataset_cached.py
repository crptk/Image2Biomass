# dataset_cached.py

import os
from typing import Optional, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from dataset import Config  # reuse your Config


class CachedBiomassDataset(Dataset):
    """
    Dataset that loads preprocessed images from disk.

    Each cache file is a numpy array saved by cache_preprocess.py:
      - shape: (H, W, C_total) where C_total = 6 (RGB + 3 veg channels), uint8

    This dataset:
      - loads the cached array
      - applies torchvision transforms on RGB only
      - resizes veg channels to match and concatenates
      - returns (image_tensor, targets) just like BiomassDataset
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Config,
        cache_dir: str,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.cache_dir = cache_dir
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        rel_path = row["image_path"]  # e.g. "train_images14/IMG_0001.jpg"
        cache_path = os.path.join(self.cache_dir, rel_path + ".npy")

        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Cached file not found: {cache_path} "
                f"(did you run cache_preprocess.py?)"
            )

        img = np.load(cache_path)  # (H, W, 6), uint8

        # --- Split RGB + vegetation ---
        rgb = img[:, :, :3].astype(np.uint8)
        extra = img[:, :, 3:].astype(np.float32) / 255.0

        rgb_pil = Image.fromarray(rgb)

        # torchvision transforms on RGB (resize/flip/jitter/normalize)
        if self.transform is not None:
            rgb_tensor = self.transform(rgb_pil)  # [3, H, W]
        else:
            rgb_tensor = transforms.ToTensor()(rgb_pil)

        # veg channels: [3, H0, W0] -> resize to config.image_size
        extra_tensor = torch.from_numpy(extra.transpose(2, 0, 1))  # [3, H0, W0]

        extra_tensor = torch.nn.functional.interpolate(
            extra_tensor.unsqueeze(0),
            size=self.config.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        img_tensor = torch.cat([rgb_tensor, extra_tensor], dim=0)  # [6, H, W]

        # --- Targets (log1p transformed) ---
        has_targets = all(col in row for col in self.config.target_cols)

        if has_targets:
            # training / validation
            targets = torch.tensor(
                [np.log1p(row[col]) for col in self.config.target_cols],
                dtype=torch.float32,
            )
            return img_tensor, targets
        else:
            # test data
            return img_tensor   # <-- IMPORTANT: return only the tensor

