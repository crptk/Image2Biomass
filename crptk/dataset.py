# dataset.py

import os
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class Config:
    """
    Central configuration for dataset / preprocessing.
    Adjust flags here or override in main.py.
    """

    # Paths (overridden in main.py)
    input_dir: str = "./csiro-biomass"
    output_dir: str = "./outputs"

    # Image size after transform
    image_size: tuple[int, int] = (224, 224)

    # 5 regression targets
    target_cols: List[str] = None

    # Weighted targets for loss + R²
    target_weights: dict = None

    # Preprocessing flags
    use_white_balance: bool = True
    use_clahe: bool = True
    use_vegetation_indices: bool = True

    # Extra channels (if enabled)
    n_vegetation_channels: int = 3  # ExG, VARI, NDI

    def __post_init__(self):
        if self.target_cols is None:
            self.target_cols = [
                "Dry_Green_g",
                "Dry_Dead_g",
                "Dry_Clover_g",
                "GDM_g",
                "Dry_Total_g",
            ]
        if self.target_weights is None:
            self.target_weights = {
                "Dry_Green_g": 0.1,
                "Dry_Dead_g": 0.1,
                "Dry_Clover_g": 0.1,
                "GDM_g": 0.2,
                "Dry_Total_g": 0.5,
            }

    @property
    def total_input_channels(self) -> int:
        """
        Total channels passed into EfficientNet.
        3 (RGB) + vegetation indices (3) = 6
        """
        n = 3
        if self.use_vegetation_indices:
            n += self.n_vegetation_channels
        return n

    def get_preprocessing_summary(self) -> str:
        enabled = []
        if self.use_white_balance:
            enabled.append("WhiteBalance")
        if self.use_clahe:
            enabled.append("CLAHE")
        if self.use_vegetation_indices:
            enabled.append("VegIdx")
        return ", ".join(enabled) if enabled else "None"


# =====================================================================
# Vegetation Indices (ExG, VARI, NDI)
# =====================================================================

class VegetationIndices:
    """Compute simple RGB-derived vegetation indices."""

    @staticmethod
    def calculate_exg(img: np.ndarray) -> np.ndarray:
        """
        Excess Green: ExG = 2G - R - B
        """
        img = img.astype(np.float32)
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        exg = 2 * g - r - b
        return np.clip(exg, 0, 255).astype(np.uint8)

    @staticmethod
    def calculate_vari(img: np.ndarray) -> np.ndarray:
        """
        VARI = (G - R) / (G + R - B)
        """
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        denom = g + r - b
        denom = np.where(denom == 0, 1, denom)

        vari = (g - r) / denom
        vari = ((vari + 1.0) * 127.5).astype(np.uint8)  # map [-1,1] -> [0,255]
        return vari

    @staticmethod
    def calculate_ndi(img: np.ndarray) -> np.ndarray:
        """
        NDI = (R - G) / (R + G)
        """
        r, g = img[:, :, 0].astype(np.float32), img[:, :, 1].astype(np.float32)

        denom = r + g
        denom = np.where(denom == 0, 1, denom)

        ndi = (r - g) / denom
        ndi = ((ndi + 1.0) * 127.5).astype(np.uint8)
        return ndi

    @staticmethod
    def calculate_all(img: np.ndarray) -> np.ndarray:
        exg = VegetationIndices.calculate_exg(img)
        vari = VegetationIndices.calculate_vari(img)
        ndi = VegetationIndices.calculate_ndi(img)
        return np.stack([exg, vari, ndi], axis=-1)  # (H, W, 3)


# =====================================================================
# Adaptive Preprocessing (White Balance + CLAHE)
# =====================================================================

class AdaptivePreprocessor:

    @staticmethod
    def white_balance(img: np.ndarray) -> np.ndarray:
        """
        Gray-world white balance normalization.
        """
        result = img.astype(np.float32)
        avg_r = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_b = np.mean(result[:, :, 2])
        avg_gray = (avg_r + avg_g + avg_b) / 3.0

        if avg_r > 0:
            result[:, :, 0] *= (avg_gray / avg_r)
        if avg_g > 0:
            result[:, :, 1] *= (avg_gray / avg_g)
        if avg_b > 0:
            result[:, :, 2] *= (avg_gray / avg_b)

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def clahe(img: np.ndarray) -> np.ndarray:
        """
        CLAHE on L channel in LAB color space.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)

        lab2 = cv2.merge([l2, a, b])
        rgb = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        return rgb


# =====================================================================
# Preprocessor Wrapper
# =====================================================================

class ImagePreprocessor:

    @staticmethod
    def preprocess(image: np.ndarray, config: Config) -> np.ndarray:
        """
        Apply preprocessing:
        1) White Balance
        2) CLAHE
        3) Vegetation indices
        Output is (H, W, C) numpy array.
        """

        # white balance
        if config.use_white_balance:
            image = AdaptivePreprocessor.white_balance(image)

        # CLAHE
        if config.use_clahe:
            image = AdaptivePreprocessor.clahe(image)

        channels = [image]  # RGB first

        # Vegetation indices
        if config.use_vegetation_indices:
            veg = VegetationIndices.calculate_all(image)
            channels.append(veg)

        # concat → final shape (H, W, C_total)
        return np.concatenate(channels, axis=-1)


# =====================================================================
# Dataset
# =====================================================================

class BiomassDataset(Dataset):
    """
    CSIRO biomass dataset after pivoting train.csv by image.

    Each __getitem__ returns:
      - image_tensor: [C, H, W]   where C = 6 (RGB + 3 vegetation indices)
      - targets:       [5]        log1p-transformed regression values
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Config,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # full path = root folder + image_path
        img_path = os.path.join(self.config.input_dir, row["image_path"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # convert BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- Apply preprocessing (returns numpy array, HWC) ---
        img = ImagePreprocessor.preprocess(img, self.config)

        # --- Split RGB + vegetation ---
        rgb = img[:, :, :3].astype(np.uint8)
        extra = img[:, :, 3:].astype(np.float32) / 255.0

        rgb_pil = Image.fromarray(rgb)

        # apply torchvision transform to RGB only
        if self.transform is not None:
            rgb_tensor = self.transform(rgb_pil)  # [3, H, W]
        else:
            rgb_tensor = transforms.ToTensor()(rgb_pil)

        # resize extra channels to match the transform size
        extra_tensor = torch.from_numpy(extra.transpose(2, 0, 1))  # [3, H0, W0]

        extra_tensor = torch.nn.functional.interpolate(
            extra_tensor.unsqueeze(0),
            size=self.config.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # final image tensor = cat(rgb, vegetation)
        img_tensor = torch.cat([rgb_tensor, extra_tensor], dim=0)

        # --- Targets (log1p transformed) ---
        targets = torch.tensor(
            [np.log1p(row[col]) for col in self.config.target_cols],
            dtype=torch.float32,
        )

        return img_tensor, targets


# =====================================================================
# Transforms
# =====================================================================

def get_transforms(config: Config, is_train=True):
    """
    Torchvision transforms for RGB part.
    Extra channels are resized separately, so no transforms there.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
