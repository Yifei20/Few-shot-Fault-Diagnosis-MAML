
"""
Prepare the CWRU dataset for training and testing. 
Pre-process the data and create a PyTorch Dataset class for the CWRU dataset 
for creating the meta-training, meta-validation, and meta-testing tasks.
"""

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms
from hst_preprocess import (
    load_HST_dataset,
    extract_dict_data,
)
from PIL import Image
import os

class HST(Dataset):

    def __init__(self, 
                 domain,
                 dir_path, 
                 transform=None):

        super(HST, self).__init__()
        if domain not in [0, 1, 2]:
            raise ValueError('Argument "domain" must be 0, 1, or 2.')
        self.domain = domain
        self.dir_path = dir_path
    
        self.img_dir = dir_path + "/WTImageData_HST/" + str(domain) + "/"
        self.img_list = os.listdir(self.img_dir)

        if transform is None:
            self.transform = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_name = self.img_list[index]
        label = torch.tensor(int(img_name.split('_')[0]), dtype=torch.int64)
        img_path = self.img_dir + img_name
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label

class HST_RAW(Dataset):

    def __init__(self, 
                 domain,
                 dir_path,
                 fft=True):
        super(HST_RAW, self).__init__()
        self.root = dir_path

        if domain not in [0, 1, 2]:
            raise ValueError('Argument "domain" must be 0, 1 or 2')
        self.domain = domain
        self.dataset = load_HST_dataset(domain, dir_path, raw=True, fft=fft)
        self.data, self.labels = extract_dict_data(self.dataset)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        sample = torch.from_numpy(sample).float()
        label = torch.tensor(label)
        return sample, label


if __name__ == '__main__':
    data = HST(1, './data')
    print(data.__getitem__(0))

