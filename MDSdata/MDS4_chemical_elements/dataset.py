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
    The label y is 0 for non-metals and 1 for metals.

    ### The data file was extracted from the publication:
    J. J. V. Ferreira, M. T. S. Pinheiro, W. R. S. dos Santos, R. da Silva Mai:
    "Graphical representation of chemical periodicity of main elements through boxplot",
    Educación Química (2016) 27, 209---216, 
    http://dx.doi.org/10.1016/j.eq.2016.04.007

    """
    df = pandas_data()
    feature_names = np.array(['atomic_radius', 'electron_affinity', 'ionization energy', 'electronegativity'])
    X = np.array(df[feature_names])
    y = np.array(df[['metallic']], dtype=int).flatten()

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
    df = pandas_data()
    # print(df)

    X, Y, features_names = data()
    atomic_radius = X[:, features_names == 'atomic_radius']
    electron_affinity = X[:, features_names == 'electron_affinity']
    print(Y)


    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    mask = Y == 0
    ax.plot(atomic_radius[mask], electron_affinity[mask], c='C0', lw=0, marker='o',  mec='C0', mfc='none', label='metallic')
    ax.plot(atomic_radius[~mask], electron_affinity[~mask], c='C1', lw=0, marker='o', mec='C1', mfc='none', label='non-meallic')
    ax.set(xlabel='atomic radius [pm]', ylabel='electron affinity [kJ/mol]')
    ax.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig('chemelem.png', pad_inches=0.1, bbox_inches='tight')

if __name__ == '__main__':
    main()