import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from lightning.pytorch import LightningModule
from torchmetrics import MeanAbsoluteError


class BiomassLightningEffNet(LightningModule):
    """
    EfficientNetV2-S with warmup + full fine-tuning for CSIRO biomass.

    Stage 1: freeze backbone → train only regression head
    Stage 2: unfreeze backbone → fine-tune entire model with lower LR
    """

    def __init__(
        self,
        num_channels: int,
        target_names: list[str],
        target_weights: dict[str, float],
        lr_head: float = 1e-4,
        lr_backbone: float = 1e-5,   # smaller LR for fine-tuning
        warmup_epochs: int = 2,      # freeze backbone for first few epochs
        pretrained: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_targets = len(target_names)
        self.target_names = target_names
        self.lr_head = lr_head
        self.lr_backbone = lr_backbone
        self.warmup_epochs = warmup_epochs

        # ----------------------------------------------------------------------
        # Weighted target tensor (same as before)
        # ----------------------------------------------------------------------
        self.register_buffer(
            "weights_tensor",
            torch.tensor([target_weights[t] for t in target_names], dtype=torch.float32),
        )

        # ----------------------------------------------------------------------
        # EfficientNetV2-S backbone
        # ----------------------------------------------------------------------
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=pretrained,
            in_chans=num_channels,
            num_classes=0,
            global_pool="avg",
        )

        # Default: freeze backbone (warmup)
        for param in self.backbone.parameters():
            param.requires_grad = False

        n_features = self.backbone.num_features

        # ----------------------------------------------------------------------
        # Regression head (same as before)
        # ----------------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_targets),
        )

        self.mae = MeanAbsoluteError()

    # ----------------------------------------------------------------------
    # Backbone unfreezing (called automatically after warmup epochs)
    # ----------------------------------------------------------------------
    def unfreeze_backbone(self):
        print("\n🔓 Unfreezing EfficientNet backbone for full fine-tuning...\n")
        for param in self.backbone.parameters():
            param.requires_grad = True

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(self, x):
        feats = self.backbone(x)
        preds = self.head(feats)
        return preds

    # ----------------------------------------------------------------------
    # Loss + R² metric
    # ----------------------------------------------------------------------
    def loss_fn(self, preds, targets):
        w = self.weights_tensor
        mse = (preds - targets) ** 2
        return (mse * w).mean()

    def weighted_r2(self, preds, targets):
        w = self.weights_tensor
        ss_res = torch.sum(w * (preds - targets) ** 2)
        y_mean = torch.sum(w * targets) / torch.sum(w)
        ss_tot = torch.sum(w * (targets - y_mean) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-8)

    # ----------------------------------------------------------------------
    # Training step
    # ----------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self(imgs)

        loss = self.loss_fn(preds, targets)
        mae = self.mae(preds, targets)
        r2 = self.weighted_r2(preds, targets)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_mae", mae, on_epoch=True)
        self.log("train_r2", r2, prog_bar=True, on_epoch=True)

        return loss

    # ----------------------------------------------------------------------
    # Validation step
    # ----------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self(imgs)

        loss = self.loss_fn(preds, targets)
        mae = self.mae(preds, targets)
        r2 = self.weighted_r2(preds, targets)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_mae", mae, on_epoch=True)
        self.log("val_r2", r2, prog_bar=True, on_epoch=True)

        return loss

    # ----------------------------------------------------------------------
    # Optimizer: separate LR for backbone and head
    # ----------------------------------------------------------------------
    def configure_optimizers(self):
        # param groups
        params = [
            {"params": self.head.parameters(), "lr": self.lr_head},
            {"params": self.backbone.parameters(), "lr": self.lr_backbone},
        ]

        opt = torch.optim.AdamW(params)

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

        return {"optimizer": opt, "lr_scheduler": sched}

    # ----------------------------------------------------------------------
    # Automatically unfreeze after warmup_epochs
    # ----------------------------------------------------------------------
    def on_train_epoch_start(self):
        if self.current_epoch == self.warmup_epochs:
            self.unfreeze_backbone()
