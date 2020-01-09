# http://www.kwangsiklee.com/2018/11/keras%EC%97%90%EC%84%9C-sequence%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%98%EC%97%AC-%EB%8C%80%EC%9A%A9%EB%9F%89-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B-%EC%B2%98%EB%A6%AC%ED%95%98%EA%B8%B0/
import numpy as np
import keras
import csv
# import cv2
import h5py

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, filename, batch_size=32, dim=(30,30,30),n_channels=1,
                 n_classes=31, shuffle=True, mode='KNU_Simplification', resizeTo=(30,30,30)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size

        label, IDs = self.parce_txt(filename)
        self.labels = label
        self.list_IDs = IDs

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.mode = mode
        self.resizeTo = resizeTo

    def get_dataset_siez(self):
        return len(self.list_IDs)

    def parce_txt(self, filename):
        f = open(filename, 'r')
        rdr = csv.reader(f)
        labels = {}
        IDs = []

        for line in rdr:
            token = line[0].split(' ')
            if len(token) > 2:
                tk = token[1]+ ' ' + token[2]
            else:
                tk = token[1]

            IDs.append(tk)
            labels[tk] = int(token[0])

        return labels,IDs

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X, X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.resizeTo, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if self.mode == 'ModelNet30':
                csv_data = np.loadtxt('Data/ModelNet30/'+ID, delimiter=',')
                data = np.reshape(csv_data,self.dim)
            elif self.mode == 'KNU_Simplification':
                csv_data = self.KNULoadTxt(ID)
                data = np.reshape(csv_data, self.dim)
            elif self.mode == 'KNU_CSV_Converted':
                csv_data = np.loadtxt(ID, delimiter=',')
                data = np.reshape(csv_data, self.dim)

            X[i,] = np.expand_dims(data, axis=3)
            # Store class
            y[i] = self.labels[ID]

        return X#, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def KNULoadTxt(self,filename):
        f = open(filename, 'r')
        lines = f.readlines()
        f.close()

        data = lines[2].split(',')
        data = list(map(int, data))
        data = np.array(data)

        return data

    # def createHDF5Dataset(self, filePath='dataset.h5'):
    #     datasetSize = len(self.labels)
    #     hdf5Dataset = h5py.File(filePath, 'w')
    #     hdf5Dataset.create_dataset(name='Voxel', shape=self.dim, dtype=int)