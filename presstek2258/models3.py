import torch
import torchvision
from lightning.pytorch import LightningModule
from torch import nn
from torch import Tensor
from torchmetrics import MeanAbsoluteError

# set torch precision
torch.set_float32_matmul_precision('high')
# clear gpu cache
torch.cuda.empty_cache()

class DeepConvNetImageOnly(LightningModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.save_hyperparameters()
        self.model = self.build_model()
        self.register_buffer('target_weights', 
                             torch.tensor([0.1,0.1,0.1,0.5,0.2]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = self.image_model(x)
        x = self.mlp(x)
        return x
    
    def loss(self, pred, target):
        residual_sos = self.target_weights * (target - pred)**2
        return residual_sos.mean()

    # training / validation loop
    def shared_step(self, mode:str, batch, batch_index):
        (image, targets, _) = batch
        x = image
        pred = self.forward(x)
        loss = self.loss(pred, targets)
        # just measure loss. if loss == 0 then r2 == 1 (perfect)
        self.log(f"{mode}_step_loss", loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def training_step(self, batch, batch_index):
        return self.shared_step('train', batch, batch_index)

    def validation_step(self, batch, batch_index):
        return self.shared_step('val', batch, batch_index)

    def test_step(self, batch, batch_index):
        return self.shared_step('test', batch, batch_index)

    def build_model(self):
        # images
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        resnet = torchvision.models.resnet18(weights=weights)
        self.image_model = nn.Sequential(
            *list(resnet.children())[:-1] # output: 512
        )
        # don't learn until the last layer (small model)
        for param in self.image_model.parameters():
            param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 5) # 5 targets
        )
