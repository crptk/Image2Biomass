# cache_preprocess.py

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dataset import Config, ImagePreprocessor  # your existing preprocessing


def cache_split(df_grouped, split_name, config, cache_root):
    """
    Caches all images from a dataframe containing image_path columns.
    split_name = "train" or "test"
    """
    print(f"\n==============================")
    print(f"📦 Caching {split_name} images...")
    print("==============================")

    for i, row in df_grouped.iterrows():
        rel_path = row["image_path"]  # e.g. train/ID00001.jpg
        src_path = Path(config.input_dir) / rel_path

        # cached path keeps structure: cached_data/train/...npy
        cache_path = cache_root / split_name / (rel_path + ".npy")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            continue  # skip if already cached

        img_bgr = cv2.imread(str(src_path))
        if img_bgr is None:
            print(f"[WARN] Missing image: {src_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # full preprocessing (white balance + CLAHE + vegetation indices)
        img_pre = ImagePreprocessor.preprocess(img_rgb, config)

        # save as numpy
        np.save(cache_path, img_pre)

        if (i + 1) % 200 == 0:
            print(f"  Cached {i + 1}/{len(df_grouped)} {split_name} images...")

    print(f"✅ Finished caching {split_name} images. Total: {len(df_grouped)}")


def main():
    this_dir = Path(__file__).resolve().parent       # /crptk
    project_root = this_dir.parent                  # /Image2Biomass
    data_root = project_root / "csiro-biomass"

    # folder to store cached data
    cache_root = this_dir / "cached_data"
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Data root:    {data_root}")
    print(f"Cache root:   {cache_root}")

    # ----------------------------------------------------------------------
    # Load configuration
    # ----------------------------------------------------------------------
    config = Config()
    config.input_dir = str(data_root)

    # ----------------------------------------------------------------------
    # Load TRAIN CSV
    # ----------------------------------------------------------------------
    train_csv_path = data_root / "train.csv"
    train_csv = pd.read_csv(train_csv_path)
    print(f"\nLoaded train.csv with {len(train_csv)} rows")

    # pivot so 1 row = 1 image
    train_grouped = train_csv.pivot_table(
        index=[
            "image_path",
            "Sampling_Date",
            "State",
            "Species",
            "Pre_GSHH_NDVI",
            "Height_Ave_cm",
        ],
        columns="target_name",
        values="target",
        aggfunc="first",
    ).reset_index()

    train_grouped = train_grouped.dropna(subset=config.target_cols)
    print(f"Unique train images: {len(train_grouped)}")

    # ----------------------------------------------------------------------
    # Load TEST CSV
    # ----------------------------------------------------------------------
    test_csv_path = data_root / "test.csv"
    test_csv = pd.read_csv(test_csv_path)
    print(f"\nLoaded test.csv with {len(test_csv)} rows")

    # The test.csv already has 1 row per test image
    test_grouped = test_csv.copy()

    # ----------------------------------------------------------------------
    # CACHE TRAIN + TEST
    # ----------------------------------------------------------------------
    cache_split(train_grouped, "train", config, cache_root)
    cache_split(test_grouped, "test", config, cache_root)

    print("\n====================================================")
    print("🎉 Finished caching ALL images!")
    print("====================================================")


if __name__ == "__main__":
    main()
