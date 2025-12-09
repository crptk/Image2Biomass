import torch
import datasets2
import models3
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import Subset
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning import Trainer
import torchvision.transforms as transforms

# get data from csvs
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# define augmentation for expanding training set
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5), # 50% chance to flip
    transforms.RandomVerticalFlip(p=0.5),   # 50% chance to flip upside down
    transforms.RandomRotation(degrees=45),  # rotate +/- 45 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # change lighting
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# define standard transformations for validation/testing
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# prepare to make the datasets
unique_image_paths = df_train['image_path'].unique()
dataset_size = len(unique_image_paths)
indices = list(range(dataset_size))
split = int(np.floor(0.7 * dataset_size))
np.random.shuffle(indices)
train_paths = unique_image_paths[indices[:split]]
val_paths = unique_image_paths[indices[split:]]
df_train_split = df_train[df_train['image_path'].isin(train_paths)]
df_val_split = df_train[df_train['image_path'].isin(val_paths)]

# 2 training datasets, one with augmentation
train_dataset = datasets2.ImageBiomassDataset(df_train_split, transform=train_transform, is_test=False)
val_dataset = datasets2.ImageBiomassDataset(df_val_split, transform=val_transform, is_test=False)
test_dataset = datasets2.ImageBiomassDataset(df_test, transform=val_transform, is_test=True)

# load the dataloaders
batch_size = 32
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=11, shuffle=True, drop_last=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=11, shuffle=False, drop_last=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=11, shuffle=False, drop_last=False)

# train the training module
model = models3.DeepConvNetImageOnly(batch_size=batch_size)
epochs = 500
logger = CSVLogger('lightning_logs', name='my_lt_module')
early_stop = EarlyStopping(
    monitor='val_step_loss',
    min_delta=0.00,
    patience=12,
    verbose=True,
    mode='min'
)
trainer = Trainer(
    logger=logger,
    max_epochs=epochs,
    log_every_n_steps=1,
    accelerator='auto',
    callbacks=[early_stop],
)
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# load the best model from checkpoints for the submission
# backpeddles 'patience' number of epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models3.DeepConvNetImageOnly.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path,
    batch_size=batch_size
)
model.to(device)

# manual testing the model
results = []
target_order = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        (image, _, image_id) = batch
        image = image.to(device)
        pred = model(image).cpu().numpy()
        for i, img_id in enumerate(image_id):
            for target_id, target_name in enumerate(target_order):
                # remake the id 
                # ex: "ID12023__Dry_Clover_g"
                # also get the prediction
                sample_id = f"{img_id}__{target_name}"
                pred_value = max(0.0, pred[i][target_id])
                results.append({'sample_id': sample_id, 'target': pred_value})

# save the results to submission.csv
submission = pd.DataFrame(results)
submission.to_csv("submission.csv", index=False)
print('submission.csv saved')
