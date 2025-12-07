import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class ImageBiomassDataset(Dataset):
    def __init__(self, dataframe, transform=None, is_test=False):
        self.transform = transform
        self.is_test = is_test
        self.target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']

        if self.is_test:
            # TEST MODE:
            self.df = dataframe.groupby('image_path').first().reset_index()
        else:
            # TRAIN MODE:
            self.df = dataframe.pivot_table(
                index=['image_path'],
                columns='target_name',
                values='target'
            ).reset_index()
            # fill missing values, if any
            self.df[self.target_cols] = self.df[self.target_cols].fillna(0.0)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB') # get image from path
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # TEST MODE:
            image_id = row['image_path'].split('/')[-1].replace('.jpg', '')
            targets = torch.zeros(5) # dummy targets, not needed
        else:
            # TRAIN MODE:
            targets = torch.tensor([float(row[col]) for col in self.target_cols], dtype=torch.float32)
            image_id = "dummy_id" # dummy id, not needed
        return image, targets, image_id
