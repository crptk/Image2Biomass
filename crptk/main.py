# main.py

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from dataset import Config, BiomassDataset, get_transforms
from dataset_cached import CachedBiomassDataset
from models import BiomassLightningEffNet


def main():
    seed_everything(42)

    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parent          # /Image2Biomass/
    data_root = project_root / "csiro-biomass"

    # ==============================================================
    # Config
    # ==============================================================
    config = Config()
    config.input_dir = str(data_root)
    config.output_dir = str(project_root / "outputs")
    # NEW: cache directory
    cache_dir = project_root / "crptk" / "cached_data"
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 70)
    print("CONFIGURATION:")
    print(f"  Input dir:    {config.input_dir}")
    print(f"  Output dir:   {config.output_dir}")
    print(f"  Cache dir:    {cache_dir}")
    print(f"  Image size:   {config.image_size}")
    print(f"  Targets:      {config.target_cols}")
    print(f"  Weights:      {config.target_weights}")
    print(f"  Preproc:      {config.get_preprocessing_summary()}")
    print(f"  Channels:     {config.total_input_channels}")
    print("=" * 70)

    # ==============================================================
    # Load CSV and pivot (same as before)
    # ==============================================================
    train_csv_path = data_root / "train.csv"
    train = pd.read_csv(train_csv_path)
    print(f"Loaded train.csv with {len(train)} rows")

    train_grouped = train.pivot_table(
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
    print(f"Unique image rows: {len(train_grouped)}")

    train_grouped = train_grouped.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df, val_df = train_test_split(
        train_grouped,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    print("----------------------------------------------------------------------")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")
    overlap = set(train_df["image_path"]).intersection(set(val_df["image_path"]))
    print(f"Overlap (should be 0): {len(overlap)}")
    print("----------------------------------------------------------------------")

    # ==============================================================
    # Choose dataset type (cached vs original)
    # ==============================================================
    # Check if we actually have cached files
    has_cache = any(str(cache_dir).endswith(".npy") for _ in [cache_dir])  # dummy

    # Better: check one file exists
    # We just test for the first image_path in the dataframe
    if len(train_grouped) > 0:
        first_path = train_grouped.iloc[0]["image_path"]
        sample_cache = cache_dir / (first_path + ".npy")
        has_cache = sample_cache.exists()

    if has_cache:
        DatasetClass = CachedBiomassDataset
        print("✅ Using CachedBiomassDataset (preprocessed images from disk).")
    else:
        DatasetClass = BiomassDataset
        print("⚠️ Cache not found, using original BiomassDataset (slower).")

    # ==============================================================
    # Datasets & DataLoaders
    # ==============================================================
    train_dataset = DatasetClass(
        train_df,
        config,
        cache_dir=str(cache_dir) if DatasetClass is CachedBiomassDataset else None,
        transform=get_transforms(config, is_train=True),
        is_train=True,
    ) if DatasetClass is CachedBiomassDataset else DatasetClass(
        train_df,
        config,
        transform=get_transforms(config, is_train=True),
        is_train=True,
    )

    val_dataset = DatasetClass(
        val_df,
        config,
        cache_dir=str(cache_dir) if DatasetClass is CachedBiomassDataset else None,
        transform=get_transforms(config, is_train=False),
        is_train=False,
    ) if DatasetClass is CachedBiomassDataset else DatasetClass(
        val_df,
        config,
        transform=get_transforms(config, is_train=False),
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,             # bump this up
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
    )

    print("DataLoaders created successfully.")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print("=" * 70)

    # ==============================================================
    # Model + Trainer (unchanged except no .to(device) calls)
    # ==============================================================
    model = BiomassLightningEffNet(
        num_channels=config.total_input_channels,
        target_names=config.target_cols,
        target_weights=config.target_weights,
        lr_head=1e-4,
        lr_backbone=1e-5,
        warmup_epochs=2,
        pretrained=True,
    )


    torch.set_float32_matmul_precision("medium")

    logger = CSVLogger(
        save_dir=str(project_root / "lightning_logs"),
        name="biomass_effnet"
    )

    callbacks = [
        EarlyStopping(monitor="val_r2", mode="max", patience=5),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = Trainer(
        max_epochs=15,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        precision="16-mixed",
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    print("Starting training...")
    print("=" * 70)

    trainer.fit(model, train_loader, val_loader)

    print("Training complete.")
    print("=" * 70)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
