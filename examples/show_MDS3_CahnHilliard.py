import matplotlib.pyplot as plt
from MDSdata import MDS3

""" Usage of the MDS-dataset 'MDS-3: Cahn-Hilliard model'

This script contains some examples for how to
use the MDS-dataset 'MDS-3: Cahn-Hilliard Model'. The images
are stored in a ZIP archive and will be extracted to a list
of numpy arrays. There are 17866 images of 64x64 pixels in size.

For further information and reference to the source of the 
data please refer to the MDS-book.
"""

def main():
    images, energies = MDS3.load_data(simulation_number=-1, return_X_y=True)
    n_images = images.shape[0]
    print("The dataset contains", n_images, "images.")
    print("They are", images.shape[1], "x", images.shape[2], " pixel in size.")
    print(f"The minimum energy is: {energies.min(): .1f}")
    print(f"The maximum energy is: {energies.max(): .1f}")
          
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 7),
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
    ax = axes.ravel()
    
    for i, idx in enumerate([10, 1000, 2000, 3000]):
        ax[i].imshow(images[idx])
        ax[i].set(title=f"image no. {idx}: E={energies[idx]:.2f}")
    plt.show()


if __name__ == '__main__':
    main()