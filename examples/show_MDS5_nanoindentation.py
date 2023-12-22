import matplotlib.pyplot as plt
from MDSdata import MDS5

""" Usage of the MDS-dataset 'MDS-5: Nanoindentation'

This script contains some examples for how to use the MDS-dataset. For 
further information and reference to the source of the data please
refer to the MDS-book.
"""

def main():
    dataset = MDS5.load_data()
    X = dataset.data
    y = dataset.target 
    print("The feature matrix has", X.shape[1], "features in columns:", dataset.feature_names)
    print(" ... and", X.shape[0], "data records as rows of X.")
    print("The class labels 0...3 of Y correspond to:", dataset.target_names)

    modulus = X[:, 0]
    hardness = X[:, 1]
    composit_type = y

    fig, ax = plt.subplots()
    ax.scatter(modulus, hardness, c=composit_type)
    ax.set(xlabel="Young's modulus [GPa]", ylabel="hardness [GPa]")
    plt.show()


    modulus, hardness = MDS5.data(material='Cr25', outlier=True)
    plt.scatter(modulus, hardness)
    plt.show()
if __name__ == '__main__':
    main()