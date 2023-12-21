import matplotlib.pyplot as plt
from MDSdata import MDS5

""" Usage of the MDS-dataset 'MDS-5: Nanoindentation'

This script contains some examples for how to use the MDS-dataset. For 
further information and reference to the source of the data please
refer to the MDS-book.
"""

def main():
    # How to use the pandas version of the dataset
    df = MDS5.data(pandas=True)
    print(df.head())

    # How to use the numpy version of the dataset
    X, feature_names = MDS5.data()
    print("number of records:", X.shape[0])
    print("feature names:    ", feature_names)

    hardness = X[:, 1]
    modulus = X[:, 0]

    fig, ax = plt.subplots()
    ax.scatter(modulus, hardness)
    ax.set(xlim=(80, 510), ylim=(0.5, 5.2))
    plt.show()


if __name__ == '__main__':
    main()