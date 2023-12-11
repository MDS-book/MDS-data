import matplotlib.pyplot as plt
from MDSdata import MDS3

""" Usage of the MDS-dataset 'MDS-3: Cahn-Hilliard model'

This script contains some examples for how to
use the MDS-dataset 'MDS-3: Cahn-Hilliard Model'. The images
are stored in a ZIP archive and will be extracted to a list
of numpy arrays. There are 5000 images of 16x16 pixels in size.

For further information and reference to the source of the 
data please refer to the MDS-book.
"""

def main():
    print(MDS3())
    images, energies = MDS3.data()
    n_images = images.shape[0]

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 7),
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
    ax = axes.ravel()
    
    for i, idx in enumerate([10, 1000, 2000, 3000]):
        ax[i].imshow(images[idx])
        ax[i].set(title=f"T={energies[idx]:.2f},")
    plt.show()


if __name__ == '__main__':
    main()