"""
PyTorch dataset classes for CWRU and HST datasets.
"""

import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms
from preprocess_cwru import load_CWRU_dataset
from preprocess_hst import load_HST_dataset
from utils import extract_dict_data
from PIL import Image
import os

class CWRU(Dataset):

    def __init__(self, 
                 domain,
                 dir_path, 
                 preprocess,
                 transform=None):

        super(CWRU, self).__init__()
        if domain not in [0, 1, 2, 3]:
            raise ValueError('Argument "domain" must be 0, 1, 2 or 3.')
        self.domain = domain
        self.dir_path = dir_path
        if preprocess != 'FFT':
            self.img_dir = dir_path + "/{}_CWRU/Drive_end_".format(preprocess) + str(domain) + "/"
        else:
            self.img_dir = dir_path + "/CWRU/Drive_end_" + str(domain) + "/"
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

class CWRU_FFT(Dataset):

    def __init__(self, 
                 domain,
                 dir_path,
                 fft=True):
        super(CWRU_FFT, self).__init__()
        self.root = dir_path

        if domain not in [0, 1, 2, 3]:
            raise ValueError('Argument "domain" must be 0, 1, 2 or 3.')
        self.domain = domain
        self.dataset = load_CWRU_dataset(domain, dir_path, raw=True, fft=fft)
        self.data, self.labels = extract_dict_data(self.dataset)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        sample = torch.from_numpy(sample).float()
        label = torch.tensor(label)
        return sample, label


class HST(Dataset):

    def __init__(self, 
                 domain,
                 dir_path, 
                 preprocess,
                 transform=None):

        super(HST, self).__init__()
        if domain not in [0, 1, 2]:
            raise ValueError('Argument "domain" must be 0, 1, or 2.')
        self.domain = domain
        self.dir_path = dir_path
    
        if preprocess == 'STFT' or preprocess == 'WT':
            self.img_dir = dir_path + "/{}_HST/".format(preprocess) + str(domain) + "/"
        elif preprocess == 'FFT':
            self.img_dir = dir_path + "/HST/".format(preprocess) + str(domain) + "/"
        else:
            raise ValueError('Invalid preprocess name.')
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

class HST_FFT(Dataset):

    def __init__(self, 
                 domain,
                 dir_path,
                 fft=True):
        super(HST_FFT, self).__init__()
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
    data = CWRU(1, './data')
    data.__getitem__(0)
    data = HST(1, './data')
    print(data.__getitem__(0))

