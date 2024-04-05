import os
import random
import numpy as np
from matplotlib import pyplot as plt


ROOT = "C:/Users/PHOENIX/Desktop/DLMI/Data"

for set in ['Train', 'Val', 'Test']:

    image = np.load(os.path.join(ROOT, set, 'MR.npy')).astype('float32')
    label = np.load(os.path.join(ROOT, set, 'CT.npy')).astype('float32')

    print()
    print(set)
    print(image.shape, label.shape)