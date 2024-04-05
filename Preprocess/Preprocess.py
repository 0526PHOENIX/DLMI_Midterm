"""
====================================================================================================
Package
====================================================================================================
"""
import os
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation, binary_erosion

import numpy as np
import nibabel as nib


"""
====================================================================================================
Global Constant
====================================================================================================
"""
RAW = "C:/Users/PHOENIX/Desktop/DLMI/Data_Raw/Val"

DATA = "C:/Users/PHOENIX/Desktop/DLMI/Data/Val"

NII = "C:/Users/PHOENIX/Desktop/DLMI/Data_Nii/Val"

PATH_LIST = [DATA, NII]


"""
====================================================================================================
Preprocess
====================================================================================================
"""
class Preprocess():

    """
    ================================================================================================
    Critical Parameters
    ================================================================================================
    """
    def __init__(self, filepath = RAW):

        # Check the Path
        for dir in PATH_LIST:
            if not os.path.exists(dir):
                os.makedirs(dir)
        
        # Load Raw Data
        self.images = np.load(os.path.join(filepath, 'MR.npy')).astype('float32')
        self.labels = np.load(os.path.join(filepath, 'CT.npy')).astype('float32')

        self.target = np.load(os.path.join(DATA, 'TG.npy')).astype('float32')

        # Check File Number
        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError('\n', 'Unequal Amount of images and labels.', '\n')
        
        self.len = self.images.shape[0]


    """
    ================================================================================================
    Main Process: Remove Background
    ================================================================================================
    """
    def main(self):

        buffer_mr = []
        buffer_ct = []
        buffer_tg = []
        for i in range(self.len):

            # Get Data
            image = self.images[i]
            label = self.labels[i]

            # Thresholding
            binary = (image > 0.0625)

            # Get Connective Component
            component, feature = ndimage.label(binary)

            # Compute Size of Each Component
            size = ndimage.sum(binary, component, range(1, feature + 1))

            # Find Largest Component
            largest_component = np.argmax(size) + 1

            mask = (component == largest_component)

            # Fill Holes in Mask
            mask = binary_dilation(mask, np.ones((15, 15)))
            mask = binary_erosion(mask, np.ones((15, 15)))
            
            # Apply Mask 
            image = np.where(mask, image, 0)
            label = np.where(mask, label, 0)
            mask = np.where(mask, 1, 0)

            # Stack
            buffer_mr.append(image)
            buffer_ct.append(label)
            buffer_tg.append(mask)

        # Get CT Series from Stack
        result_mr = np.stack(buffer_mr, axis = 0)
        result_ct = np.stack(buffer_ct, axis = 0)
        result_tg = np.stack(buffer_tg, axis = 0)

        # Save Numpy Data
        np.save(os.path.join(DATA, 'MR.npy'), result_mr)
        np.save(os.path.join(DATA, 'CT.npy'), result_ct)
        np.save(os.path.join(DATA, 'TG.npy'), result_tg)

        # Check Progress
        print()
        print('Done')
        print()
        print('===================================================================================')

        return
    
    """
    ================================================================================================
    Check Statistics
    ================================================================================================
    """
    def check(self):

        for i in range(self.len):

            image = self.images[i]
            label = self.labels[i]

            space = "{: <15.2f}\t{: <15.2f}"
            print(i + 1, 'MR:')
            print(space.format(image.max(), image.min()))
            print(space.format(image.mean(), image.std()))
            print()
            print(i + 1, 'CT:')
            print(space.format(label.max(), label.min()))
            print(space.format(label.mean(), label.std()))
            print()
            print('===============================================================================')

        return

    """
    ================================================================================================
    Convert .npy to .nii
    ================================================================================================
    """
    def npy2nii(self):

        # Save Nifti Data
        images = nib.Nifti1Image(self.images, np.eye(4))
        nib.save(images, os.path.join(NII, 'MR.nii'))

        labels = nib.Nifti1Image(self.labels, np.eye(4))
        nib.save(labels, os.path.join(NII, 'CT.nii'))

        target = nib.Nifti1Image(self.target, np.eye(4))
        nib.save(target, os.path.join(NII, 'TG.nii'))

        # Check Progress
        print()
        print('Done')
        print()
        print('===================================================================================')

        return
    

"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':
    
    # pre = Preprocess(RAW)
    # pre.main()

    pre = Preprocess(DATA)
    pre.npy2nii()

    # pre.check()