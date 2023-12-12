import matplotlib.pyplot as plt
from MDSdata import MDS2

""" Usage of the MDS-dataset 'MDS-2: Ising model'

This script contains some examples for how to
use the MDS-dataset 'MDS-2: Ising Model'. The images
are stored in a ZIP archive and will be extracted to a list
of numpy arrays. There are 5000 images of 64x64 pixels in size.

For further information and reference to the source of the 
data please refer to the MDS-book.
"""

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