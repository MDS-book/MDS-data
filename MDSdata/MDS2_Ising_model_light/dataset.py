import numpy as np
import os
from os.path import join

import matplotlib.pyplot as plt
from numpy import ndarray
from MDSdata.io import get_Ising_images_temperatures_and_labels


"""
Two files are required: a zip archiv that contains a number of
images without any directory, and a csv file that contains three
columns with the names (as first row):
filenames,temperatures,labels

The filenames must correspond to the the names in the zip archive.
The labels (0 or 1) indicate if the temperature is below (0) or 
above the Curie temperature (1).
"""

# The absolute path is required when importing this package! Otherwise
# a wrong relative path is resolved and reading a file from within this
# script does not work properly. You can see this with
# `print(os.path.dirname(os.path.abspath(__file__)))`
p1 = join(os.path.dirname(os.path.abspath(__file__)), 'images_16.zip')
p2 = join(os.path.dirname(os.path.abspath(__file__)), 'labels_16.csv')


class MDS2_light:
    n_images = 5000
    pixels = (16, 16)

    def __init__(self) -> None:
        # self.class_name = self.__class__.__name__
        pass

    @staticmethod
    def data(verbose=False) -> (ndarray, list, list):
        images, temperatures, labels = \
            get_Ising_images_temperatures_and_labels(
                zip_filename=p1, 
                csv_filename=p2,
                verbose=verbose
            )
        
        # just some sanity checks
        assert images.shape[0] == MDS2_light.n_images, f"The zip file is expected to contain 5000 images. Actual value: {images.shape[0]}"
        assert images[0].shape == MDS2_light.pixels, f"The images in the zip file should be {MDS2_light.pixels}  in size"
        return images, temperatures, labels
    
    def __str__(self):
        s = 'MDS2 (light) dataset obtained from simulations ' + \
            f'with the Ising model. The dataset contains {MDS2_light.n_images} ' + \
            'images sampled from the temperature range [0, 2Tc]. ' + \
            f'The image are {MDS2_light.pixels[0]} x {MDS2_light.pixels[1]} pixels in size.'
        return s



def main():
    images, temperatures, labels = MDS2_light.data()

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 7),
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
    ax = axes.ravel()
    
    for i, idx in enumerate([10, 1500, 3000, 4500]):
        ax[i].imshow(images[idx])
        ax[i].set(title=f"T={temperatures[idx]:.2f},  label={labels[idx]}")
    plt.show()


if __name__ == '__main__':
    main()