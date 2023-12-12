import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from MDSdata import MDS1



def main():
    strain_T0, strain_T400, strain_T600, \
        stress_T0, stress_T400, stress_T600 = MDS1.data()
    
    fig, ax = plt.subplots()
    ax.scatter(strain_T0, stress_T0, marker='.', label='0°C')
    ax.scatter(strain_T400, stress_T400, marker='.', label='400°C')
    ax.scatter(strain_T600, stress_T600, marker='.', label='600°C')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()