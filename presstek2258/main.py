import torch
import datasets
import models
import pandas as pd
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from lightning.pytorch.loggers import CSVLogger
from lightning import Trainer
import torchvision.transforms as transforms
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


# image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# make the datasets
dataset = datasets.ImageBiomassDataset(df_train, transform=transform)
#test_dataset = datasets.ImageBiomassDataset(df_test, transform=transform)
batch_size = 32
train_dataset, val_dataset = random_split(
    dataset,
    (0.7, 0.3)
)

# load the dataloaders
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=11, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=11, shuffle=False)
#test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=11, shuffle=False)


# setup training
model = models.DeepConvNet(learning_rate=1e-5)  # Lower learning rate for stable training
epochs = 30  # Increased epochs for better training with scheduler
logger = CSVLogger('lightning_logs', name='my_lt_module')

# improvements for learning 
early_stop = EarlyStopping(
    monitor='val_step_loss',
    min_delta=0.00,
    patience=8,
    verbose=False,
    mode='min'
)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# train the module
trainer = Trainer(
    logger=logger, 
    max_epochs=epochs, 
    log_every_n_steps=1, 
    accelerator='auto',
    callbacks=[early_stop, lr_monitor]
)
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
#trainer = Trainer(logger=logger)
#trainer.test(model=model, dataloaders=test_dataloader)
