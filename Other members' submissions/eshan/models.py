import torch
from torch import nn
from lightning.pytorch import LightningModule
import torchvision.models as tv
import torchmetrics
import torch.nn.functional as F

class BiomassNet(LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # 1. RESNET50 BACKBONE
        backbone = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        self.image_model = backbone
        image_out_dim = 2048

        # 2. ATTRIBUTE MLP
        self.attribute_model = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU()
        )

        # 3. REGRESSION HEAD 
        self.regressor = nn.Sequential(
            nn.Linear(image_out_dim + 32, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 5)    # predict 5 biomass components
        )

        # METRICS
        self.mae = torchmetrics.MeanAbsoluteError()
        self.r2 = torchmetrics.R2Score(multioutput="uniform_average")

        # Will store predictions during testing
        self.test_outputs = []

    def forward(self, image, attrs):
        img_feat = self.image_model(image)
        attr_feat = self.attribute_model(attrs)
        combined = torch.cat([img_feat, attr_feat], dim=1)
        return self.regressor(combined)

    def training_step(self, batch, idx):
        image, attrs, targets = batch
        preds = self(image, attrs)

        loss = F.mse_loss(preds, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, idx):
        image, attrs, targets = batch
        preds = self(image, attrs)

        loss = F.mse_loss(preds, targets)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", self.mae(preds, targets))
        self.log("val_r2", self.r2(preds, targets), prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def on_test_epoch_start(self):
        """Initialize storage for predictions."""
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        image, attrs, _ = batch
        preds = self(image, attrs)

        # store manually for later
        self.test_outputs.append(preds.detach().cpu())

        return preds

    def on_test_epoch_end(self):
        """Collect all predictions at the end of testing."""
        self.test_preds = torch.cat(self.test_outputs, dim=0)
        print("Collected test predictions:", self.test_preds.shape)

