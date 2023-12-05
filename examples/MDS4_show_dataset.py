import matplotlib.pyplot as plt
from MDSdata import MDS4

""" Usage of the MDS-dataset 'MDS-4: Chemical Elements'

This script contains some examples for how to
use the MDS-dataset 'MDS-4: Chemical Elements'. For further
information and reference to the source of the data please
refer to the MDS-book.
"""

def main():
    # How to use the numpy version of the dataset
    X, Y, feature_names = MDS4.data()
    print(X)
    print(Y)
    print(feature_names)


    # How to use the pandas version of the dataset
    df = MDS4.data(pandas=True)

    plt.scatter(x=df['atomic_radius'], 
                y=df['electronegativity'],
                c=df['metallic'])
    plt.show()


if __name__ == '__main__':
    main()