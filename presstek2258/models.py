import torch
from typing import Tuple
from lightning.pytorch import LightningModule
from torch import nn
from torch import Tensor
from torchmetrics import MeanSquaredError, MeanAbsoluteError

# set torch precision
torch.set_float32_matmul_precision('high')
# clear gpu cache
torch.cuda.empty_cache()

class DeepConvNet(LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.mae = MeanAbsoluteError()
        self.learning_rate = learning_rate
        self.save_hyperparameters()  # Save hyperparameters for checkpoints
        self.model = self.build_model()

    def configure_optimizers(self):
        # optimizer uses scheduler to reduce learning_rate on plateaus
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_step_loss'
        }

    def forward(self, x):
        (image, attributes) = x
        image_features = self.image_model(image)
        image_features = torch.flatten(image_features, 1) # to fix adaptive pooling
        attr_features = self.attribute_model(attributes)
        combined_features = torch.cat((image_features, attr_features), dim=1)
        output = self.combined_model(combined_features)
        return output.flatten()

    def loss(self, pred, target):
        return nn.functional.mse_loss(pred, target)

    # training / validation loop
    def shared_step(self, mode:str, batch:Tuple[Tensor, Tensor], batch_index:int):
        (image, attributes, target_name, target) = batch
        x = (image, attributes)

        pred = self.forward(x)
        loss = self.loss(pred, target) # already uses mse
        mae = self.mae(pred, target)

        # setups the weights for r2 score
        #
        # Dry_Clover_g = 0.1
        # Dry_Dead_g = 0.1
        # Dry_Green_g = 0.1
        # Dry_Total_g = 0.5
        # GDM_g = 0.2
        weights = torch.zeros_like(target, dtype=torch.float32)
        weights[target_name.int() == 0] = 0.1
        weights[target_name.int() == 1] = 0.1
        weights[target_name.int() == 2] = 0.1
        weights[target_name.int() == 3] = 0.5
        weights[target_name.int() == 4] = 0.2

        # calculate the r2 score
        # residual sum of squares = residual_sos
        # total sum of squares = total_sos
        weighted_mean = torch.sum(weights * target) / torch.sum(weights)
        residual_sos = torch.sum(weights * (target - pred)**2)
        total_sos = torch.sum(weights * (target - weighted_mean)**2)
        weighted_r2 = 1 - (residual_sos / total_sos)

        self.log(f"{mode}_step_mae", mae , prog_bar=False)
        self.log(f"{mode}_step_score", weighted_r2 , prog_bar=True)
        self.log(f"{mode}_step_loss", loss, prog_bar=False)
        return loss

    def training_step(self, batch, batch_index):
        return self.shared_step('train', batch, batch_index)

    def validation_step(self, batch, batch_index):
        return self.shared_step('val', batch, batch_index)

    def test_step(self, batch, batch_index):
        return self.shared_step('test', batch, batch_index)

    # the scheduler steps based on the validation loss
    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:  # only one trainer at a time
            scheduler = self.lr_schedulers()
            if scheduler is not None:
                val_loss = self.trainer.callback_metrics.get('val_step_loss')
                if val_loss is not None:
                    scheduler.step(val_loss)

    def build_model(self):
        self.image_model = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # adaptive pooling
        )
        self.attribute_model = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.combined_model = nn.Sequential(
            nn.Linear(128*4*4 + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
