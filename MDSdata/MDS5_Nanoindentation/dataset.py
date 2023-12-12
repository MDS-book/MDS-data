import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join

# The absolute path is required when importing this package! Otherwise
# a wrong relative path is resolved and reading a file from within this
# script does not work properly. You can see this with
# `print(os.path.dirname(os.path.abspath(__file__)))`
p = join(os.path.dirname(os.path.abspath(__file__)), 'CuCr.csv')  




"""
The data file is part of the supplementary data of the publication of:
=================================
Chen Zhang, Clémence Bos, Stefan Sandfeld, Ruth Schwaiger, 
Unsupervised Learning of Nanoindentation Data to Infer Microstructural Details of Complex Materials,
https://arxiv.org/abs/2309.06613

Description
===========
Tabular data for 664 nanoindents of a CuCr60 metallic material


Usage:

from MDSbook_utils.MDScode.Datasets.chemical_element_data import dataset

df, features = dataset.pandas_data()
print(features)
display(df)


X = dataset.numpy_data()

"""

def pandas_data():
    """Returns a dataframe with the data    
    """
    df = pd.read_csv(p)
    feature_names = ['Modulus', 'Hardness']
    return df[feature_names]


def numpy_data():
    df = pandas_data()
    feature_names = ['Modulus', 'Hardness']
    X = np.array(df[feature_names])

    return X, feature_names


def data(pandas=False):
    """MDS-dataset 'MDS-5: Nanoindentation CuCr60 (hardness and modulus)'.
    
    Returns the two features and the class mapping (column names)
    for CuCr60.

    :params pandas: whether to return numpy or pandas data
        pandas=False: Returns input matrix X, and the class mapping
        pandas=True: Returns DataFrame with data matrix and the column names

    ### The data file was taken from the supplementary material of the publication:
    Chen Zhang, Clémence Bos, Stefan Sandfeld, Ruth Schwaiger, 
    Unsupervised Learning of Nanoindentation Data to Infer Microstructural Details of Complex Materials,
    https://arxiv.org/abs/2309.06613

    It can also be found at https://doi.org/10.5281/zenodo.8336072

    Please reference these publications if you are using the dataset or methods for your own research.
    """    
    if pandas:
        return pandas_data()
    else:
        return numpy_data()


def main():
    df = pandas_data()
    print(df)
    data, feature_names = numpy_data()
    hardness = data[:, 0]
    modulus = data[:, 1]

    plt.scatter(hardness, modulus)
    plt.show()

if __name__ == '__main__':
    main()