import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join


# The absolute path is required when importing this package! Otherwise
# a wrong relative path is resolved and reading a file from within this
# script does not work properly. You can see this with
# `print(os.path.dirname(os.path.abspath(__file__)))`
p1 = join(os.path.dirname(os.path.abspath(__file__)), 'strain.csv')  
p2 = join(os.path.dirname(os.path.abspath(__file__)), 'stress.csv')


def pandas_data():
    """Returns a dataframe with the data    
    """
    dfstrain = pd.read_csv(p1)
    dfstress = pd.read_csv(p2)
    #df.columns = features
    return dfstrain, dfstress


def numpy_data():
    df_strain, df_stress = pandas_data()
    #feature_names = ['atomic_radius', 'electron_affinity', 'ionization energy', 'electronegativity']
    
    columns = df_strain.columns  # virst column label is empty (=index)
    strain_T0 = np.array(df_strain[columns[1]])
    strain_T400 = np.array(df_strain[columns[2]])
    strain_T600 = np.array(df_strain[columns[3]])
    stress_T0   = np.array(df_stress[columns[1]])
    stress_T400 = np.array(df_stress[columns[2]])
    stress_T600 = np.array(df_stress[columns[3]])

    return strain_T0, strain_T400, strain_T600, stress_T0, stress_T400, stress_T600


def data(pandas=False):
    """MDS-dataset 'MDS-1: Tensile Test'.

    if pandas=False: returns 3 strains and 3 stresses at T=0, 400, 600°C
    if pandas=True:  returns a DataFrame with strain and one with stress
    """    
    if pandas:
        return pandas_data()
    else:
        return numpy_data()
    



def main():
    print(data(pandas=True))


if __name__ == '__main__':
    main()