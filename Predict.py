import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import losses
import keras.backend as K

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from DataGenerator import DataGenerator
from Model import build_model

def VisualizeVolxe(data, threshold):
    if(len(data.shape) > 3):
        data = np.squeeze(data, axis=0)
        data = np.squeeze(data, axis=3)

    data = data > threshold

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')

    ax.voxels(data, edgecolor="k")

    plt.show()

def ConstructModel():
    K.clear_session()

    model = build_model(input_voxel_size=[30,30,30],
                        augmenting_dropout_rate=.5,
                        conv_num_filters=[64, 256, 256, 64, 1],
                        conv_filter_sizes=[9,4, 5, 6],
                        conv_strides=[3,2, 2, 3],
                        desc_dims=[6912,6912])

    model.load_weights('CKPT/Epoch_10.h5')

    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9)

    model.compile(loss='binary_crossentropy',optimizer=sgd)

    return model

def MakePrediction(model, input):
    output = model.predict(input)

    return output

def LoadSingleInput(filename):
    #csv_data = np.loadtxt('Data/ModelNet30/bathtub_te/bathtub_te_1.csv'delimiter=',')
    csv_data = np.loadtxt(filename, delimiter = ',')
    data = np.reshape(csv_data, (30,30,30))
    data = np.expand_dims(data, axis=0)
    data = np.expand_dims(data, axis=4)
    return data

def main(argv):
    model = ConstructModel()
    print(model.summary())

    input = LoadSingleInput('Data/ModelNet30/bathtub_te/bathtub_te_1.csv')
    output = MakePrediction(model, input)

    VisualizeVolxe(input, 0.5)
    VisualizeVolxe(output, 0.5)

if __name__== '__main__':
    main(sys.argv)