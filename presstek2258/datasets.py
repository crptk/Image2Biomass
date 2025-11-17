import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class ImageBiomassDataset(Dataset):
    # change all data to numerical except images
    def data_to_numerical(self, dataframe):
        dataframe['sample_id'] = pd.Categorical(dataframe['sample_id']).codes
        dataframe['Sampling_Date'] = pd.to_datetime(dataframe['Sampling_Date'])
        dataframe['Sampling_Date'] = (dataframe['Sampling_Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
        dataframe['State'] = pd.Categorical(dataframe['State']).codes
        dataframe['Species'] = pd.Categorical(dataframe['Species']).codes
        dataframe['target_name'] = pd.Categorical(dataframe['target_name']).codes
        return dataframe
    
    def __init__(self, dataframe, transform=None):
        self.df = self.data_to_numerical(dataframe)
        self.transform = transform
        self.att_classes = ['sample_id', 'Sampling_Date', 'State', 'Species', 
                       'Pre_GSHH_NDVI', 'Height_Ave_cm']
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB') # get image from path
        if self.transform:
            image = self.transform(image)
        attributes = torch.tensor([row[col] for col in self.att_classes], dtype=torch.float32)
        target_name = torch.tensor(row['target_name'], dtype=torch.float32)
        target = torch.tensor(row['target'], dtype=torch.float32)
        
        return image, attributes, target_name, target

