import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join

# The absolute path is required when importing this package! Otherwise
# a wrong relative path is resolved and reading a file from within this
# script does not work properly. You can see this with
# `print(os.path.dirname(os.path.abspath(__file__)))`
p = join(os.path.dirname(os.path.abspath(__file__)), 'element_properties.csv')  




"""
The data file was extracted from:
=================================
J. J. V. Ferreira, M. T. S. Pinheiro, W. R. S. dos Santos, R. da Silva Mai:
"Graphical representation of chemical periodicity of main elements through boxplot",
Educación Química (2016) 27, 209---216
http://dx.doi.org/10.1016/j.eq.2016.04.007

Description
===========
Tabular data for chosen properties of a number of metallic and non-metallic elements


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
    #df.columns = features
    return df #, df.columns


def numpy_data():
    """MDS-dataset 'MDS-4: Chemical Elements'.
    
    Returns the input matrix X, output vector y and the class mapping
    for chosen properties of a number of metallic and non-metallic elements.

    ### The data file was extracted from the publication:
    J. J. V. Ferreira, M. T. S. Pinheiro, W. R. S. dos Santos, R. da Silva Mai:
    "Graphical representation of chemical periodicity of main elements through boxplot",
    Educación Química (2016) 27, 209---216, 
    http://dx.doi.org/10.1016/j.eq.2016.04.007

    """
    df = pandas_data()
    feature_names = ['atomic_radius', 'electron_affinity', 'ionization energy', 'electronegativity']
    X = np.array(df[feature_names])
    y = np.array(df[['metallic']]).flatten()

    return X, y, feature_names


def data(pandas=False):
    """MDS-dataset 'MDS-4: Chemical Elements'.
    

    Returns the features, labels and the class mapping
    for chosen properties of a number of metallic and non-metallic elements.

    :params pandas: whether to return numpy or pandas data
        pandas=False: Returns input matrix X, output vector y and the class mapping
        pandas=True: Returns DataFrame with data matrix and the column names

    ### The data file was extracted from the publication:
    J. J. V. Ferreira, M. T. S. Pinheiro, W. R. S. dos Santos, R. da Silva Mai:
    "Graphical representation of chemical periodicity of main elements through boxplot",
    Educación Química (2016) 27, 209---216, 
    http://dx.doi.org/10.1016/j.eq.2016.04.007

    """    
    if pandas:
        return pandas_data()
    else:
        return numpy_data()


def main():
    df, features = pandas_data()
    print(df)
    print(features)
    data = numpy_data()
    density = data[:, 0]
    hardness = data[:, 1]

    plt.scatter(density, hardness)
    #plt.show()

if __name__ == '__main__':
    main()