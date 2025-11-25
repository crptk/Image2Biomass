import torch
import os
import database
import models
import pandas as pd
from torch.utils.data import random_split, DataLoader
from lightning.pytorch.loggers import CSVLogger
from lightning import Trainer
import torchvision.transforms as transforms
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor


def main():

    df_train = pd.read_csv("csiro-biomass/train.csv")
    df_test = pd.read_csv("csiro-biomass/test.csv")

    # image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # make the datasets
    dataset = database.ImageBiomassDataset(df_train, transform=transform)
    test_dataset = database.ImageBiomassDataset(df_test, transform=transform)
    batch_size = 32
    
    train_len = int(0.7 * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    # load dataloaders (num_workers > 0 requires main guard!)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=7,
        shuffle=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=7,
        shuffle=False,
        persistent_workers=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        num_workers=7, 
        shuffle=False
    )


    # setup training
    model = models.DeepConvNet(learning_rate=1e-5)
    epochs = 5
    logger = CSVLogger('lightning_logs', name='my_lt_module')

    early_stop = EarlyStopping(
        monitor='val_step_loss',
        min_delta=0.00,
        patience=8,
        verbose=False,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    #model = models.DeepConvNet.load_from_checkpoint(
    #"lightning_logs/version_12/checkpoints/epoch=29-step=XXXX.ckpt"
    #)

    trainer = Trainer(
        logger=logger,
        max_epochs=epochs,
        log_every_n_steps=1,
        accelerator='auto',
        callbacks=[early_stop, lr_monitor]
    )

    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    
    trainer.test(model=model, dataloaders=test_dataloader)


if __name__ == '__main__':
    main()
