"""
====================================================================================================
Package
====================================================================================================
"""
import os
import random
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset


"""
====================================================================================================
Training
====================================================================================================
"""
class Train(Dataset):

    def __init__(self, root = ""):

        # Filepath
        self.root = root
        self.images_path = os.path.join(self.root, 'Train', 'MR.npy')
        self.labels_path = os.path.join(self.root, 'Train', 'CT.npy')

        # Load MR Data: (570, 256, 256)
        self.images = np.load(self.images_path).astype('float32')
        self.images = torch.from_numpy(self.images)

        # Load CT Data: (570, 256, 256)
        self.labels = np.load(self.labels_path).astype('float32')
        self.labels = torch.from_numpy(self.labels)

        # Check Data Quantity
        if self.images.shape != self.labels.shape:
            raise ValueError('Unequal Amount of Images and Labels.')

    def __len__(self):
        
        return self.images.shape[0]

    def __getitem__(self, index):

        # Load MR Data: (1, 256, 256)
        image = self.images[index : index + 1, :, :]
        
        # Load CT Data: (1, 256, 256)
        label = self.labels[index : index + 1, :, :]

        return (image, label)
    

"""
====================================================================================================
Validation
====================================================================================================
"""
class Val(Dataset):

    def __init__(self, root = ""):

        # Filepath
        self.root = root
        self.images_path = os.path.join(self.root, 'Val', 'MR.npy')
        self.labels_path = os.path.join(self.root, 'Val', 'CT.npy')

        # Load MR Data: (90, 256, 256)
        self.images = np.load(self.images_path).astype('float32')
        self.images = torch.from_numpy(self.images)

        # Load CT Data: (90, 256, 256)
        self.labels = np.load(self.labels_path).astype('float32')
        self.labels = torch.from_numpy(self.labels)

        # Check Data Quantity
        if self.images.shape != self.labels.shape:
            raise ValueError('Unequal Amount of Images and Labels.')

    def __len__(self):
        
        return self.images.shape[0]

    def __getitem__(self, index):

        # Load MR Data: (1, 256, 256)
        image = self.images[index : index + 1, :, :]
        
        # Load CT Data: (1, 256, 256)
        label = self.labels[index : index + 1, :, :]

        return (image, label)
    

"""
====================================================================================================
Testing
====================================================================================================
"""
class Test(Dataset):

    def __init__(self, root = ""):

        # Filepath
        self.root = root
        self.images_path = os.path.join(self.root, 'Test', 'MR.npy')
        self.labels_path = os.path.join(self.root, 'Test', 'CT.npy')

        # Load MR Data: (150, 256, 256)
        self.images = np.load(self.images_path).astype('float32')
        self.images = torch.from_numpy(self.images)

        # Load CT Data: (150, 256, 256)
        self.labels = np.load(self.labels_path).astype('float32')
        self.labels = torch.from_numpy(self.labels)

        # Check Data Quantity
        if self.images.shape != self.labels.shape:
            raise ValueError('Unequal Amount of Images and Labels.')

    def __len__(self):
        
        return self.images.shape[0]

    def __getitem__(self, index):

        # Load MR Data: (1, 256, 256)
        image = self.images[index : index + 1, :, :]
        
        # Load CT Data: (1, 256, 256)
        label = self.labels[index : index + 1, :, :]

        return (image, label)
    

"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    filepath = "/home/ccy/DLMI/Data"

    train = Train(filepath)
    val = Val(filepath)
    test = Test(filepath)

    for i in range(5):

        index = random.randint(0, len(train) - 1)

        image, label = train[index]
        
        print()
        print('image:')
        print(image.min(), image.max())
        print(image.mean(), image.std())
        print('label:')
        print(label.min(), label.max())
        print(label.mean(), label.std())
        print()