import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os

class ImageBiomassDataset(Dataset):

    TARGET_COLS = [
        "Dry_Green_g",
        "Dry_Dead_g",
        "Dry_Clover_g",
        "GDM_g",
        "Dry_Total_g"
    ]

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.copy()
        self.transform = transform

        # Attributes
        self.att_classes = [
            'sample_id', 'Sampling_Date', 'State', 'Species',
            'Pre_GSHH_NDVI', 'Height_Ave_cm'
        ]

        # Ensure every attribute exists (test.csv misses some)
        for col in self.att_classes:
            if col not in self.df.columns:
                self.df[col] = 0

        # Ensure target columns exist (test.csv => filled with zeros)
        for col in self.TARGET_COLS:
            if col not in self.df.columns:
                self.df[col] = 0.0

        # Convert data
        self.df = self.data_to_numerical(self.df)

    def data_to_numerical(self, df):
        # sample_id categorical
        df['sample_id'] = pd.Categorical(df['sample_id']).codes

        # Date conversion
        if 'Sampling_Date' in df.columns:
            try:
                df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])
                df['Sampling_Date'] = (
                    df['Sampling_Date'] - pd.Timestamp("1970-01-01")
                ) // pd.Timedelta('1D')
            except:
                df['Sampling_Date'] = 0

        # Categorical conversions
        for col in ['State', 'Species']:
            if col in df.columns:
                df[col] = pd.Categorical(df[col]).codes
            else:
                df[col] = 0

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        img_path = os.path.join("csiro-biomass/", row['image_path'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Attributes vector
        attributes = torch.tensor(
            [row[col] for col in self.att_classes],
            dtype=torch.float32
        )

        # 5-regression targets
        targets = torch.tensor(
            [row[col] for col in self.TARGET_COLS],
            dtype=torch.float32
        )

        return image, attributes, targets
