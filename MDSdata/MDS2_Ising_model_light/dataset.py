import numpy as np
import os
from os.path import join

import matplotlib.pyplot as plt
from MDSdata.io import get_images_temperatures_and_labels


"""
Two files are required: a zip archiv that contains a number of
images without any directory, and a csv file that contains the three
columns with the names (as first row):
filenames,temperatures,labels

The labels tell if the temperature is below (0) or above the Curie temperature (1)
"""

# The absolute path is required when importing this package! Otherwise
# a wrong relative path is resolved and reading a file from within this
# script does not work properly. You can see this with
# `print(os.path.dirname(os.path.abspath(__file__)))`
p = join(os.path.dirname(os.path.abspath(__file__)), '16.zip')
p2 = join(os.path.dirname(os.path.abspath(__file__)), 'labels_16.csv')




def main():
    images, temperatures, labels = get_images_temperatures_and_labels(
        zip_filename=p, 
        csv_filename=p2
    )



    plt.imshow(images[1])
    plt.show()


if __name__ == '__main__':
    main()