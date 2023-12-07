import numpy as np
import os
from os.path import join

import matplotlib.pyplot as plt
from MDSdata.io import get_images_temperatures_and_labels


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
p = join(os.path.dirname(os.path.abspath(__file__)), '16.zip')
p2 = join(os.path.dirname(os.path.abspath(__file__)), 'labels_16.csv')



def data():
    images, temperatures, labels = \
        get_images_temperatures_and_labels(
            zip_filename=p, 
            csv_filename=p2
        )
    return images, temperatures, labels


def main():
    images, temperatures, labels = data()

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 7),
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
    ax = axes.ravel()
    
    for i, idx in enumerate([10, 1500, 3000, 4500]):
        ax[i].imshow(images[idx])
        ax[i].set(title=f"T={temperatures[idx]:.2f},  label={labels[idx]}")
    plt.show()


if __name__ == '__main__':
    main()