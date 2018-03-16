"""
Dataset

(c) Aleksei Tiulpin, University of Oulu, 2018
"""

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pandas as pd
import cv2
import os
import io
from tqdm import tqdm
import torchvision.transforms as transforms

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

class InvasiveSpeciesDataset(data.Dataset):
    def __init__(self, dataset_loc, split, transform):
        # dataset location
        self.dataset_loc = dataset_loc
        # Train data split
        self.split = split
        # Augmentations and other transformations
        self.transforms = transform

    def __getitem__(self, idx):
        entry = self.split.iloc[idx]
        fname, label = entry['name'], entry['invasive']
        img = cv2.imread(os.path.join(self.dataset_loc, str(fname)+'.jpg'))
        if img is None:
            print(fname, idx, entry)
        img = self.transforms(img)

        return {'img': img, 'label':label}
    
    def __len__(self):
        return self.split.shape[0]
        
