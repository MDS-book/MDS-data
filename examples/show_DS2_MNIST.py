import matplotlib.pyplot as plt
import numpy as np
from mdsdata import DS2



def main():
    img, lbl = DS2().load_data(return_X_y=True, train=True)
    
    print("number of images:", img.shape[0])
    print("number of classes for training data: ", end='')
    print(np.histogram(lbl, np.arange(-0.5, 10.5, 1))[0])


    fig, axes = plt.subplots(ncols=5, nrows=3, figsize=(7, 5))
    rng = np.random.default_rng()
    for ax in axes.ravel():
        idx = rng.integers(img.shape[0])
        ax.imshow(img[idx], cmap='gray', vmin=0, vmax=255)
        ax.set(xticks=[], yticks=[], title=f"{lbl[idx]}")
    plt.show()



if __name__ == '__main__':
    main()
