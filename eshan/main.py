import torch
import os
import database
import models
import pandas as pd
from torch.utils.data import random_split, DataLoader
from lightning.pytorch.loggers import CSVLogger
from lightning import Trainer
import torchvision.transforms as transforms
from lightning.pytorch.callbacks import (
    EarlyStopping, LearningRateMonitor, ModelCheckpoint
)

def main():

    # CHECKPOINT DIRECTORY
    checkpoint_dir = "csiro-biomass/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # LOAD DATA
    df_train = pd.read_csv("csiro-biomass/train.csv")
    df_test = pd.read_csv("csiro-biomass/test.csv")

    # TRANSFORMS
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # DATASETS
    dataset = database.ImageBiomassDataset(df_train, transform=transform)
    test_dataset = database.ImageBiomassDataset(df_test, transform=transform)

    # TRAIN/VAL SPLIT
    train_len = int(0.7 * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    batch_size = 32

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=7
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=7
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=7
    )

    # MODEL
    model = models.BiomassNet(learning_rate=1e-4)

    # LOGGING / CALLBACKS
    logger = CSVLogger('lightning_logs', name='biomass_regression')

    early_stop = EarlyStopping(
        monitor='val_loss', patience=8, mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # CHECKPOINT SAVING
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,           
        filename="best",                   
        save_top_k=1,                      
        monitor="val_loss",                
        mode="min"
    )

    trainer = Trainer(
        logger=logger,
        max_epochs=5,
        accelerator='auto',
        log_every_n_steps=1,
        callbacks=[early_stop, lr_monitor, checkpoint_callback]
    )

    # TRAIN
    trainer.fit(model, train_dataloader, val_dataloader)

    # TEST
    trainer.test(model, dataloaders=test_dataloader)

if __name__ == '__main__':
    main()
