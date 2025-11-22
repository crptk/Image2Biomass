
import os
from pathlib import Path
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from dataset_cached import CachedBiomassDataset
from dataset import Config, get_transforms
from models import BiomassLightningEffNet


def run_fold(fold_idx, train_df, val_df, config, cache_dir, project_root):
    print("\n" + "="*80)
    print(f"🚀 Starting Fold {fold_idx+1}")
    print("="*80)

    # ========================================================
    # Dataset / DataLoaders
    # ========================================================
    train_dataset = CachedBiomassDataset(
        train_df,
        config,
        cache_dir=str(cache_dir),
        transform=get_transforms(config, is_train=True),
        is_train=True,
    )

    val_dataset = CachedBiomassDataset(
        val_df,
        config,
        cache_dir=str(cache_dir),
        transform=get_transforms(config, is_train=False),
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    # ========================================================
    # Model
    # ========================================================
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

    # ========================================================
    # Callbacks / Logging / Checkpoints
    # ========================================================
    ckpt_dir = project_root / "cv_folds" / f"fold_{fold_idx+1}"
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best",
        monitor="val_r2",
        mode="max",
        save_top_k=1
    )

    callbacks = [
        checkpoint,
        EarlyStopping(monitor="val_r2", patience=5, mode="max"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = Trainer(
        max_epochs=15,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    # ========================================================
    # Train
    # ========================================================
    trainer.fit(model, train_loader, val_loader)

    # return best score
    best_r2 = checkpoint.best_model_score.item()
    print(f"\n🎯 Fold {fold_idx+1} R² = {best_r2:.6f}\n")

    return best_r2


def main():
    seed_everything(42)

    # ========================================================
    # Paths
    # ========================================================
    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parent
    data_root = project_root / "csiro-biomass"
    cache_dir = this_dir / "cached_data"

    # ========================================================
    # Load config + data
    # ========================================================
    config = Config()
    config.input_dir = str(data_root)

    train_csv = pd.read_csv(data_root / "train.csv")

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

    print(f"Total usable images: {len(train_grouped)}")

    # ========================================================
    # 5-Fold Split
    # ========================================================
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_grouped)):
        train_df = train_grouped.iloc[train_idx].reset_index(drop=True)
        val_df = train_grouped.iloc[val_idx].reset_index(drop=True)

        fold_r2 = run_fold(fold_idx, train_df, val_df, config, cache_dir, project_root)
        fold_scores.append(fold_r2)

    # ========================================================
    # Final CV Result
    # ========================================================
    print("\n" + "="*80)
    print("🔥 5-FOLD CROSS VALIDATION COMPLETE")
    print("="*80)

    for i, score in enumerate(fold_scores):
        print(f"Fold {i+1}: R² = {score:.6f}")

    print("\n📈 Average R²:", np.mean(fold_scores))
    print("📉 Std R²:", np.std(fold_scores))
    print("="*80)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
