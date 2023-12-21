import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join




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



class MDS5:
    """MDS-dataset 'MDS-5: Nanoindentation CuCr60 (hardness and modulus)'."""  
        
    # The absolute path is required when importing this package! Otherwise
    # a wrong relative path is resolved and reading a file from within this
    # script does not work properly. You can see this with
    # `print(os.path.dirname(os.path.abspath(__file__)))`
    p0 = join(os.path.dirname(os.path.abspath(__file__)), 'Cr0_full.csv')  
    p25 = join(os.path.dirname(os.path.abspath(__file__)), 'Cr25_full.csv')
    p60 = join(os.path.dirname(os.path.abspath(__file__)), 'Cr60_full.csv')
    p100 = join(os.path.dirname(os.path.abspath(__file__)), 'Cr100_full.csv') 

    p0red = join(os.path.dirname(os.path.abspath(__file__)), 'Cr0_reduced.csv')  
    p25red = join(os.path.dirname(os.path.abspath(__file__)), 'Cr25_reduced.csv')
    p60red = join(os.path.dirname(os.path.abspath(__file__)), 'Cr60_reduced.csv')
    p100red = join(os.path.dirname(os.path.abspath(__file__)), 'Cr100_reduced.csv') 

    def __init__(self) -> None:
        pass

    @staticmethod
    def data(material='all', raw=False):
        """MDS-dataset 'MDS-5: Nanoindentation CuCr60 (hardness and modulus)'.
        
        Returns the either E and H for a particular composit or for all materials.

        ### The data file was taken from the supplementary material of the publication:
        Chen Zhang, Clémence Bos, Stefan Sandfeld, Ruth Schwaiger, 
        Unsupervised Learning of Nanoindentation Data to Infer Microstructural Details of Complex Materials,
        https://arxiv.org/abs/2309.06613

        It can also be found at https://doi.org/10.5281/zenodo.8336072

        Please reference these publications if you are using the dataset or methods for your own research.

        :param material: either 'all' or the material name (Cu or Cr0, Cr25, Cr60, Cr or Cr100)
        :param raw: True: the full dataset will be read; False: outliers are removed
        :returns: two 1D ndarrays containing YOung's modulus E and hardness H
                  Only if material='all' then a list of tuples (E, H) is returned:
                  [(E_Cr0, H_Cr0), ..., (E_Cr100, H_Cr100)]
        """
        if material != 'all':
            if material == 'Cu' or material == 'Cr0':
                df = pd.read_csv(MDS5.p0 if raw else MDS5.p0red)
            elif material == 'Cr25':
                df = pd.read_csv(MDS5.p25 if raw else MDS5.p25red)
            elif material == 'Cr60':
                df = pd.read_csv(MDS5.p60 if raw else MDS5.p60red)
            elif material == 'Cr' or material == 'Cr100':
                df = pd.read_csv(MDS5.p100 if raw else MDS5.p100red)
            
            return np.array(df['E']), np.array(df['H'])
        
        else:
            # Return all datasets:
            modulus, hardness, class_names, class_id = [], [], [], []

            if raw:
                paths = [MDS5.p0, MDS5.p25, MDS5.p60, MDS5.p100]
            else:
                paths = [MDS5.p0red, MDS5.p25red, MDS5.p60red, MDS5.p100red]
            label_names = ['0% Cr', '25% Cr', '60% Cr', '100% Cr']

            for i, p in enumerate(paths):
                df = pd.read_csv(p)
                modulus += df['E'].tolist()
                hardness += df['H'].tolist()
                class_names.append(label_names[i])
                class_id += df['E'].size * [i]

            X = np.stack((modulus, hardness), axis=0).T
            Y = np.array(class_id, dtype=int)
            return X, Y, class_names


def main():

    X, Y, class_names = MDS5.data()
    print("feature matrix has the shape:", X.shape)
    print("The class label 0...3 correspond to the materials:", class_names)

    modulus = X[:, 0]
    hardness = X[:, 1]
    plt.scatter(modulus, hardness, c=Y, cmap='Paired_r')
    plt.show()

    #return 
    fig, ax = plt.subplots(ncols=2, figsize=(9, 4))

    modulus, hardness = MDS5.data(material='Cr0', raw=True)
    ax[0].scatter(modulus, hardness, alpha=0.5)

    modulus, hardness = MDS5.data(material='Cr25', raw=True)
    ax[0].scatter(modulus, hardness, alpha=0.5)

    modulus, hardness = MDS5.data(material='Cr60', raw=True)
    ax[0].scatter(modulus, hardness, alpha=0.5)

    modulus, hardness = MDS5.data(material='Cr100', raw=True)
    ax[0].scatter(modulus, hardness, alpha=0.5)

    # ------------------------------------------------
    # list_of_E_and_H = MDS5.data(raw=False)
    # materials = ['0% Cr', '25% Cr', '60% Cr', '100% Cr']
    # for i, (modulus, hardness) in enumerate(list_of_E_and_H):
    #     print(i)
    #     ax[1].scatter(modulus, hardness, alpha=0.5, c=f'C{i}', label=materials[i])
    # ax[1].legend()
    
    plt.show()

    #print(np.array(list_of_E_and_H).shape)

if __name__ == '__main__':
    main()