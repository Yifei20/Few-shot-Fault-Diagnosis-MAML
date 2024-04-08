
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
from PIL import Image
import os

class CWRU(Dataset):

    def __init__(self, 
                 domain,
                 dir_path, 
                 transform=None):

        super(CWRU, self).__init__()
        if domain not in [0, 1, 2, 3]:
            raise ValueError('Argument "domain" must be 0, 1, 2 or 3.')
        self.domain = domain
        self.dir_path = dir_path
    
        self.img_dir = dir_path + "/STFTImageData/Drive_end_" + str(domain) + "/"
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


if __name__ == '__main__':
    data = CWRU(1, './data')
    data.__getitem__(0)
