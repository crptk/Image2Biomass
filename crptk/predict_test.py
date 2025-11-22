import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from dataset_cached import CachedBiomassDataset
from dataset import Config, get_transforms
from models import BiomassLightningEffNet


@torch.no_grad()
def main():
    print("\n========================================================")
    print("🚀 Generating Kaggle Submission...")
    print("========================================================\n")

    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parent
    data_root = project_root / "csiro-biomass"
    cache_dir = this_dir / "cached_data"

    # ======================================
    # Load test.csv (5 rows per image!)
    # ======================================
    raw_test = pd.read_csv(data_root / "test.csv")

    # unique images only
    test_df = raw_test[["image_path"]].drop_duplicates().reset_index(drop=True)

    print(f"Loaded test.csv with {len(test_df)} unique test images")

    # ======================================
    # Config
    # ======================================
    config = Config()
    config.input_dir = str(data_root)

    # ======================================
    # Dataset + Loader
    # ======================================
    test_dataset = CachedBiomassDataset(
        df=test_df,
        config=config,
        cache_dir=str(cache_dir),
        transform=get_transforms(config, is_train=False),
        is_train=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    # ======================================
    # Load fold checkpoints
    # ======================================
    fold_paths = []
    for i in range(1, 6):
        path = project_root / "cv_folds" / f"fold_{i}" / "best.ckpt"
        if not path.exists():
            raise FileNotFoundError(f"❌ Missing checkpoint: {path}")
        fold_paths.append(path)


    print("\n🔍 Found fold checkpoints:")
    for p in fold_paths:
        print("  •", p)

    # ======================================
    # Inference across all folds
    # ======================================
    print("\n🤖 Running inference across 5 folds...\n")

    all_fold_preds = []

    for fold_idx, ckpt_path in enumerate(fold_paths):
        print(f"Loading Fold {fold_idx+1} model...")

        model = BiomassLightningEffNet.load_from_checkpoint(
            checkpoint_path=str(ckpt_path), strict=False
        )
        model.eval()
        model.cuda()

        preds_list = []

        for batch in test_loader:
            imgs = batch.cuda(non_blocking=True)
            preds = model(imgs)   # log1p predictions
            preds_list.append(preds.cpu().numpy())

        preds_fold = np.concatenate(preds_list, axis=0)
        all_fold_preds.append(preds_fold)

        print(f"✔ Fold {fold_idx+1} predictions complete.")

    # ======================================
    # Ensemble: average predictions
    # ======================================
    print("\n📊 Ensembling all folds...")
    all_fold_preds = np.stack(all_fold_preds, axis=0)    # [5, N, 5]
    ensemble_preds_log = all_fold_preds.mean(axis=0)     # [N, 5]

    # ======================================
    # Inverse log1p
    # ======================================
    print("\n🔄 Applying inverse log1p...")
    ensemble_preds = np.expm1(ensemble_preds_log)

    # ======================================
    # Build Kaggle submission
    # ======================================
    print("\n🧩 Building submission DataFrame...")

    # Kaggle requires:
    # sample_id,target
    # IDxxxx__Dry_Clover_g,123.4
    target_order = [
        "Dry_Green_g",
        "Dry_Dead_g",
        "Dry_Clover_g",
        "GDM_g",
        "Dry_Total_g",
    ]

    sub_rows = []

    for i, row in raw_test.iterrows():
        img_path = row["image_path"]
        target_name = row["target_name"]
        sample_id = row["sample_id"]

        # find index of this image in test_df
        img_index = test_df.index[test_df["image_path"] == img_path][0]

        # find which target index to read from prediction vector
        target_idx = target_order.index(target_name)

        pred_value = ensemble_preds[img_index, target_idx]

        sub_rows.append([sample_id, pred_value])

    submission = pd.DataFrame(sub_rows, columns=["sample_id", "target"])

    # ======================================
    # Save CSV
    # ======================================
    out_path = project_root / "submission.csv"
    submission.to_csv(out_path, index=False)

    print("\n========================================================")
    print("🎉 Submission file saved!")
    print(f"📁 Path: {out_path}")
    print("========================================================")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
