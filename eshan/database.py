import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os


class ImageBiomassDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.copy()
        self.transform = transform

        # Attributes your model expects
        self.att_classes = [
            'sample_id', 'Sampling_Date', 'State', 'Species',
            'Pre_GSHH_NDVI', 'Height_Ave_cm'
        ]

        # Create missing columns for test.csv
        for col in self.att_classes:
            if col not in self.df.columns:
                self.df[col] = 0   # default value for test data

        # Convert columns that DO exist
        self.df = self.data_to_numerical(self.df)

    def data_to_numerical(self, dataframe):
        # sample_id → categorical
        dataframe['sample_id'] = pd.Categorical(dataframe['sample_id']).codes

        # Sampling_Date exists only in training
        if 'Sampling_Date' in dataframe.columns:
            try:
                dataframe['Sampling_Date'] = pd.to_datetime(dataframe['Sampling_Date'])
                dataframe['Sampling_Date'] = (
                    dataframe['Sampling_Date'] - pd.Timestamp("1970-01-01")
                ) // pd.Timedelta('1D')
            except:
                dataframe['Sampling_Date'] = 0   # fallback for test
        else:
            dataframe['Sampling_Date'] = 0

        # Convert categorical fields that exist
        for col in ['State', 'Species', 'target_name']:
            if col in dataframe.columns:
                dataframe[col] = pd.Categorical(dataframe[col]).codes
            else:
                dataframe[col] = 0

        return dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = os.path.join("csiro-biomass/", row['image_path'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Collect attributes vector
        attributes = torch.tensor(
            [row[col] for col in self.att_classes],
            dtype=torch.float32
        )

        target_name = torch.tensor(row['target_name'], dtype=torch.float32)

        # `target` does NOT exist in test.csv → provide dummy
        target = torch.tensor(row['target'], dtype=torch.float32) if 'target' in row else torch.tensor(0.0)

        return image, attributes, target_name, target

    

