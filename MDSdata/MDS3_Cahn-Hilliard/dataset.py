import numpy as np
import os
from os.path import join

import matplotlib.pyplot as plt
from numpy import ndarray
from MDSdata.io import get_Ising_images_temperatures_and_labels


import zipfile, io
from PIL import Image
import numpy as np
import os.path
import pandas as pd
from tqdm import tqdm



"""
Dataset MDS 3
Two files are required: a zip archiv that contains a number of
images without any directory, and a csv file that contains three
columns with the names (as first row):
filenames,temperatures,labels

The filenames must correspond to the the names in the zip archive.
The labels (0 or 1) indicate if the temperature is below (0) or 
above the Curie temperature (1).
"""

# The absolute path is required when importing this package! Otherwise
# a wrong relative path is resolved and reading a file from within this
# script does not work properly. You can see this with
# `print(os.path.dirname(os.path.abspath(__file__)))`
p = join(os.path.dirname(os.path.abspath(__file__)), '')
p1 = join(os.path.dirname(os.path.abspath(__file__)), 'CH_datasets_adaptive_part_A.zip')
p2 = join(os.path.dirname(os.path.abspath(__file__)), 'train_data_partA_and_B_run_from_0_17.csv')





class MDS3:
    n_images = 5000
    pixels = (16, 16)

    def __init__(self) -> None:
        # self.class_name = self.__class__.__name__
        pass

    @staticmethod
    def data(verbose=False) -> (ndarray, list, list):
        images, temperatures, labels = \
            get_Ising_images_temperatures_and_labels(
                zip_filename=p1, 
                csv_filename=p2,
                verbose=verbose
            )
        
        # just some sanity checks
        # assert images.shape[0] == MDS2_light.n_images, f"The zip file is expected to contain 5000 images. Actual value: {images.shape[0]}"
        # assert images[0].shape == MDS2_light.pixels, f"The images in the zip file should be {MDS3.pixels}  in size"
        return images, temperatures, labels
    
    # def __str__(self):
    #     s = 'MDS2 (light) dataset obtained from simulations ' + \
    #         f'with the Ising model. The dataset contains {MDS2_light.n_images} ' + \
    #         'images sampled from the temperature range [0, 2Tc]. ' + \
    #         f'The image are {MDS2_light.pixels[0]} x {MDS2_light.pixels[1]} pixels in size.'
    #     return s


def get_X_array(train_csv, train_data_prepath):
    X_train_2D = []
    for im_name in train_csv.filename:
        im_path = train_data_prepath + im_name
        img_arr = np.array(Image.open(im_path).convert('L'))
        X_train_2D.append(img_arr)

    X_train_2D = np.stack((X_train_2D), axis=0)
    num_images_train, image_size = np.shape(X_train_2D)[0], np.shape(X_train_2D)[1]

    # flatten the image arrays
    X_train = X_train_2D.reshape(num_images_train, image_size ** 2 )

    # if scale:
    #     scaler_X = StandardScaler()
    #     X_train = scaler_X.fit_transform(X_train)
    # else:
    #     pass

    return X_train


def get_images_and_energies_from_directory(run_num, csv_filename, base_dir):
    """
    :param run_num: number of the simulation run (there are altogether 17)
    """
    csv_dataset = pd.read_csv(csv_filename)

    images = []
    energies = []
    for i, row in csv_dataset.iterrows():
        if run_num != row['run_num']: continue
        # print("index:", i, ',  file:', row['filename'], '  energy=', row['energy'], '  num_run=', row['run_num'])

        im_path = join(p, row['filename'])
        img_arr = np.array(Image.open(im_path).convert('L'))
        images.append(img_arr)
        energies.append(row['energy'])

    images = np.array(images, dtype=float)
    return images, energies


def main():
    csv_filename = p2
    zip_filename = p1

    for run_num in range(18):  # 18 und 19 sind leer!!!!!!
        print(run_num)
        images, energies = get_images_and_energies_from_directory(run_num=run_num, csv_filename=csv_filename, base_dir=p)
        n_images = images.shape[0]

        #------------------------------------------------------------------------------------------
        print('Saving generated data')
        filenames = []
        zip_filename = join(os.path.dirname(os.path.abspath(__file__)), f"images_{run_num}.zip")
        progress_bar = tqdm(range(n_images), total=n_images, leave=False)

        # === create a zip file with all image-PNGs ===
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for index in progress_bar:
                filename = f'{str(index).zfill(5)}.png'
                filenames.append(filename)
                with zipf.open(filename, 'w') as zfile:
                    img = Image.fromarray(images[index, :, :].astype('uint8'), mode='L')
                    img.save(zfile, format='PNG')


        df = pd.DataFrame()
        df['filenames'] = filenames
        df['energy'] = energies
        # df['labels'] = np.array(labels)
        #df.to_csv(os.path.join(path, f'labels_{N}.csv'), index=False)
        df.to_csv(join(os.path.dirname(os.path.abspath(__file__)), f"labels_{run_num}.csv"), index=False)
        #------------------------------------------------------------------------------------------






    plt.imshow(images[10])
    plt.show()

    return


    csv_dataset = pd.read_csv(csv_filename)
    print()
    print(csv_dataset.head())
    
    images = []
    energies = []
    for i, row in csv_dataset.iterrows():
        #print("index:", i, ',  data:', *row)
        run_num = row['run_num']
        if run_num != 7: continue

        print("index:", i, ',  file:', row['filename'], '  energy=', row['energy'], '  num_run=', row['run_num'])
        im_path = join(p, row['filename'])
        img_arr = np.array(Image.open(im_path).convert('L'))
        images.append(img_arr)
        energies.append(row['energy'])

        if i==100: break

    print(csv_dataset.columns)

    # get_X_array(csv_dataset, p)


    """[
        'CH_datasets_adaptive_part_A/', 
        'CH_datasets_adaptive_part_A/energy_2.txt', 
        'CH_datasets_adaptive_part_A/cahn_hilliard_0/', 
        'CH_datasets_adaptive_part_A/cahn_hilliard_0/sxx_00356.png',
    """

    # with zipfile.ZipFile(zip_filename) as myzip:
    #     filenames = myzip.namelist()
    #     print(filenames[:100])
    #     return

    #     with myzip.open(filenames[0]) as myfile:
    #         img_data = myfile.read()
    #         buf = io.BytesIO(img_data)
    #         img = np.asarray(Image.open(buf))


    # images, temperatures, labels = MDS3.data()

    # fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 7),
    #                          gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
    # ax = axes.ravel()
    
    # for i, idx in enumerate([10, 1500, 3000, 4500]):
    #     ax[i].imshow(images[idx])
    #     ax[i].set(title=f"T={temperatures[idx]:.2f},  label={labels[idx]}")
    # plt.show()


if __name__ == '__main__':
    main()