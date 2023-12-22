import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join



# The class `Bunch`` is copied from the Python package scikit-learn, 
# version 1.3.2, https://scikit-learn.org/.
# The original code is licensed under the BSD license.
# The BSD license text is contained in the file 
# `MDSdata/BSD_3-Clause_License` of this repository/package.
# The purpose of copying that class was to allow the same interface 
# as scikit-learn.
class Bunch(dict):
    """Container object exposing keys as attributes.

    Bunch objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`.

    Examples
    --------
    >>> from sklearn.utils import Bunch
    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass




DESCR = \
    """
MDS-dataset MDS-5: Nanoindentation CuCr60 (hardness and modulus)
----------------------------------------------------------------
        
**Dataset Characteristics:**

    :Number of Instances: 378 (.., .., .., .. for each of four classes)
    :Number of Attributes: 2 numeric, predictive attributes and the class
    :Attribute Information:
        - Young's modulus in GPa
        - indentation hardness in GPA
        - class:
                - '0% Cr'
                - '25% Cr'
                - '60% Cr'
                - '100% Cr'

    :Summary Statistics:

    :Missing Attribute Values: None

    :Class Distribution: ... for each of 4 classes.

    :Creator: Chen Zhang, Clémence Bos, Stefan Sandfeld, Ruth Schwaiger

    :Donor:

    :date July, 2023 

    
    Nanoindentation of four different Cu/Cr composites. The dataset has two 
    features, the Young's modulus E and the hardness H, both of which are 
    given in GPa.

    The data file is part of the supplementary material of the publication:
    Chen Zhang, Clémence Bos, Stefan Sandfeld, Ruth Schwaiger: "Unsupervised 
    Learning of Nanoindentation Data to Infer Microstructural Details of 
    Complex Materials, https://arxiv.org/abs/2309.06613 which can also be 
    found at https://doi.org/10.5281/zenodo.8336072

    Please reference these publications if you are using the dataset or methods
    for your own research.

    By default, outliers are removed, and the total number of data records is 
    the above given number. If the outlier (e.g., for the purpose of an 
    exercise) are required, then the above given total number of records and 
    samples per class will be different ones.             
    """


class MDS5:
    """MDS-dataset 'MDS-5: Nanoindentation CuCr60 (hardness and modulus)'.
    
    The interface of the `data` method has been designed to conform closely
    with the well-established interface of scikit-learn (see 
    https://scikit-learn.org/stable/datasets.html). The main difference is
    that when features and targets are returned as pandas DataFrame, they
     are always returned as separate obejcts and never as combined DataFrame.
    """  
        
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
    def load_data(outlier=False, return_X_y=False, as_frame=False):
        """MDS-dataset 'MDS-5: Nanoindentation CuCr60 (hardness and modulus)'.
        
        Nanoindentation of four different Cu/Cr composites. The dataset has two
        features, the Young's modulus E and the hardness H, both of which are 
        given in GPa.

        =================   ==============
        Classes                          4
        Samples per class      ??/??/??/??
        Records total                  938
        Dimensionality                   2
        Features            real, positive
        =================   ==============

        More details can be found in the MDS book and at https://MDS-book.org

        The data file is part of the supplementary material of the publication:
        Chen Zhang, Clémence Bos, Stefan Sandfeld, Ruth Schwaiger: "Unsupervised 
        Learning of Nanoindentation Data to Infer Microstructural Details of 
        Complex Materials, https://arxiv.org/abs/2309.06613 which can also be 
        found at https://doi.org/10.5281/zenodo.8336072

        Please reference these publications if you are using the dataset or methods
        for your own research.
        
        Parameters
        ----------
        return_X_y : bool, default=False
            If True, returns ``(data, target)`` instead of a 
            dictionary-like Bunch
            ``{data, target, taget_names, DESCR, feature_names}``. 

        as_frame : bool, default=False
            If True, the data is a pandas DataFrames, and the target is a 
            pandas DataFrame or Series depending on the number of target 
            columns.

        outlier: bool, default=False
            In the default case, outliers are removed, and the total number of
            data records is the above given number. If the outlier (e.g., 
            for the purpose of an exercise) are required, then the above
            given total number of records and samples per class are different.


        Returns
        -------
        data : Either a set of feature matrix and target vector or a 
               Bunch (i.e., a dictionary that can be accessed using a dot)
               with the following attributes:

            data : {ndarray, dataframe} of shape (???, 2)
                The data matrix. If `as_frame=True`, `data` will be a pandas
                DataFrame.
            target: {ndarray, Series} of shape (150,)
                The classification target. If `as_frame=True`, `target` will be
                a pandas Series.
            feature_names: list
                The names of the dataset columns.
            target_names: list
                The names of target classes.
            frame: DataFrame of shape (???, 3)
                Only present when `as_frame=True`. DataFrame with `data` and
                `target`.

            DESCR: str
                The full description of the dataset.
        """

        _feature_names = ["Young's modulus", "hardness"]
        label_names = ['0% Cr', '25% Cr', '60% Cr', '100% Cr']
        modulus, hardness, class_id = [], [], []

        if outlier:
            paths = [MDS5.p0, MDS5.p25, MDS5.p60, MDS5.p100]
        else:
            paths = [MDS5.p0red, MDS5.p25red, MDS5.p60red, MDS5.p100red]

        for i, p in enumerate(paths):
            df = pd.read_csv(p)
            modulus += df['E'].tolist()
            hardness += df['H'].tolist()
            class_id += df['E'].size * [i]
        
        X = np.stack((modulus, hardness), axis=0).T
        y = np.array(class_id, dtype=int)

        if as_frame:
            X = pd.DataFrame(data=X, columns=_feature_names)
            y = pd.DataFrame(data=y, columns=['target'])

        if return_X_y:
            return X, y

        return Bunch(
            data=X, 
            target=y,
            feature_names=_feature_names, 
            target_names=label_names, 
            DESCR=DESCR,
        )



def main():
    dataset = MDS5.load_data()
    X = dataset.data
    y = dataset.target 

    print("The feature matrix has the shape:", X.shape)
    print("The two features are:", dataset.feature_names)
    print("The class label 0...3 correspond to the materials:", dataset.target_names)

    
    X, y = MDS5.load_data(return_X_y=True)


    return

    X, Y, feature_names, label_names = MDS5.data()
    print("The feature matrix has the shape:", X.shape)
    print("The class label 0...3 correspond to the materials:", label_names)

    modulus = X[:, 0]
    hardness = X[:, 1]
    plt.scatter(modulus, hardness, c=Y, cmap='Paired_r')
    plt.show()

    #return 
    fig, ax = plt.subplots(ncols=2, figsize=(9, 4))

    modulus, hardness, _, _ = MDS5.data(material='Cr0', outlier=True)
    ax[0].scatter(modulus, hardness, alpha=0.5)

    modulus, hardness, _, _ = MDS5.data(material='Cr25', outlier=True)
    ax[0].scatter(modulus, hardness, alpha=0.5)

    modulus, hardness, _, _ = MDS5.data(material='Cr60', outlier=True)
    ax[0].scatter(modulus, hardness, alpha=0.5)

    modulus, hardness, _, _ = MDS5.data(material='Cr100', outlier=True)
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