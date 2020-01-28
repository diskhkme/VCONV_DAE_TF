# http://www.kwangsiklee.com/2018/11/keras%EC%97%90%EC%84%9C-sequence%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%98%EC%97%AC-%EB%8C%80%EC%9A%A9%EB%9F%89-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B-%EC%B2%98%EB%A6%AC%ED%95%98%EA%B8%B0/
import numpy as np
import pandas as pd
import keras
import csv
# import cv2
import h5py
import os
from scipy.ndimage import zoom

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=32, dim=(30,30,30),n_channels=1,
                 n_classes=31, shuffle=True, mode='KNU_Simplification',
                 load_data_into_memory=True,
                 load_HDF_path='train_dataset.h5'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.mode = mode
        self.load_data_into_memory = load_data_into_memory
        self.load_HDF_path = load_HDF_path
        self.data = []
        self.HDF_data = []

        if os.path.exists(load_HDF_path):
            self.HDF_data = h5py.File(self.load_HDF_path, 'r')
            self.numData = self.HDF_data['Voxel'].size

            if self.load_data_into_memory:
                for i in range(self.numData):
                    data = self.HDF_data['Voxel'][i].reshape(self.dim)
                    data = data.astype(int)
                    self.data.append(np.expand_dims(data, axis=3))

                    # self.data.append(self.HDF_data['Voxel'][i])
                    if i % 1000 == 0:
                        print("{0} / {1}".format(i, self.numData))

            self.data = np.array(self.data)

        self.on_epoch_end()

    def get_dataset_siez(self):
        return self.numData

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.numData / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        if self.load_data_into_memory == True:
            X = self.data[indexes,]
        else:
            # Generate data
            X = self.__data_generation(indexes)

        return X, X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.numData)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.dim[2], self.n_channels),dtype=int)

        # Generate data
        for i in range(self.batch_size):
            data = self.HDF_data['Voxel'][indexes[i]].reshape(self.dim)
            data = data.astype(int)

            X[i,] = np.expand_dims(data, axis=3)

        return X
