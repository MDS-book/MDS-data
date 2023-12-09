import numpy as np
import os
from os.path import join

import matplotlib.pyplot as plt
from numpy import ndarray
from MDSdata.io import get_Ising_images_temperatures_and_labels, read_images_from_zip_archive, get_CahnHilliard_images_and_energies


import zipfile, io
from PIL import Image
import numpy as np
import os.path
import pandas as pd
from tqdm import tqdm



"""
Dataset MDS 3 -- Cahn-Hilliard

The whole dataset consists of 18 simulations. For each,
two files are required: a zip archiv that contains a number of
images without any directory, and a csv file that contains three
columns with the names (as first row):
filenames,energy
The filenames must correspond to the the names in the zip archive.
"""

# The absolute path is required when importing this package! Otherwise
# a wrong relative path is resolved and reading a file from within this
# script does not work properly. You can see this with
# `print(os.path.dirname(os.path.abspath(__file__)))`
p = join(os.path.dirname(os.path.abspath(__file__)), '')


class MDS3:
    pixels = (64, 64)

    def __init__(self) -> None:
        pass

    @staticmethod
    def data(simulation_number=-1, verbose=False) -> (ndarray, list, list):
        """Reads and returns images and energie values for the Cahn-Hilliard datatset.

        :param simulation_number: if given (as int or list of ints), then only these 
            simulations will be read. Otherwise, all 18 simulations will be read.
        
        """
        assert isinstance(simulation_number, (int, list)), \
            "simulation_number must be either an int or a list of ints"
        
        if isinstance(simulation_number, int):
            simulation_number = [simulation_number]
        if simulation_number == [-1]:
            simulation_number = range(18)
        
        all_images = []
        all_energies = []

        progress_bar = tqdm(simulation_number, total=len(simulation_number), leave=False)
        for n in progress_bar:
            zip_file = join(p, f"images_{n}.zip")
            csv_file = join(p, f"labels_{n}.csv")

            images, energies = get_CahnHilliard_images_and_energies(zip_file, csv_file, False)
            all_images += images.tolist()
            all_energies += energies
            
        all_images = np.array(all_images, dtype=float)
        
        # just some sanity checks
        assert all_images[0].shape == MDS3.pixels, f"The images in the zip files should be {MDS3.pixels}  in size"

        return all_images, all_energies
    
    # def __str__(self):
    #     s = 'MDS2 (light) dataset obtained from simulations ' + \
    #         f'with the Ising model. The dataset contains {MDS2_light.n_images} ' + \
    #         'images sampled from the temperature range [0, 2Tc]. ' + \
    #         f'The image are {MDS2_light.pixels[0]} x {MDS2_light.pixels[1]} pixels in size.'
    #     return s



def main():

    images, energies = MDS3.data(simulation_number=1)

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 7),
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
    ax = axes.ravel()
    
    for i, idx in enumerate([10, 500, 1000, 1500]):
        ax[i].imshow(images[idx])
        ax[i].set(title=f"T={energies[idx]:.2f}")
    plt.show()


if __name__ == '__main__':
    main()