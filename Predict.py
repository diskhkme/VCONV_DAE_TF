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
import cv2

from DataGenerator import DataGenerator
from Model import build_model
import GlobalParam as _g

def PrintVoxel(data, filepath):
    data = data.reshape((_g.INPUT_VOXEL_SIZE[0]*_g.INPUT_VOXEL_SIZE[1], _g.INPUT_VOXEL_SIZE[2]))
    np.savetxt(filepath,data,delimiter=',')

def VisualizeVolxel(data):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')

    ax.voxels(data, edgecolor="k")

    plt.show()

def SqueezeAndBinarize(data, threshold):
    if(len(data.shape) > 3):
        data = np.squeeze(data, axis=0)
        data = np.squeeze(data, axis=3)

    # cv2.imshow('',data[:,:,5])
    data = data > threshold

    return data

def ConstructModel():
    K.clear_session()

    model = build_model(input_voxel_size=_g.INPUT_VOXEL_SIZE,
                        augmenting_dropout_rate=_g.AUGMENTED_DROPOUT_RATE,
                        conv_num_filters=_g.CONV_NUM_FILTERS,
                        conv_filter_sizes=_g.CONV_FILTER_SIZES,
                        conv_strides=_g.CONV_STRIDES,
                        desc_dims=_g.DESC_DIMS)

    model.load_weights(_g.TRAIN_OUTPUT_FOLDER + _g.WEIGHT_FILE_NAME)

    sgd = optimizers.SGD(lr=_g.LEARNING_RATE, decay=_g.DECAY, momentum=_g.MOMENTUM)

    model.compile(loss='binary_crossentropy',optimizer=sgd)

    return model

def MakePrediction(model, input):
    output = model.predict(input)

    return output

def LoadSingleInput(filename):
    #csv_data = np.loadtxt('Data/ModelNet30/bathtub_te/bathtub_te_1.csv'delimiter=',')
    csv_data = np.loadtxt(filename, delimiter = ',')
    data = np.reshape(csv_data, (_g.INPUT_VOXEL_SIZE[0], _g.INPUT_VOXEL_SIZE[1],_g.INPUT_VOXEL_SIZE[2]))
    data = np.expand_dims(data, axis=0)
    data = np.expand_dims(data, axis=4)
    return data

def main(argv):
    model = ConstructModel()
    print(model.summary())

    input = LoadSingleInput(_g.PREDICTION_FILE_PATH)
    output = MakePrediction(model, input)

    input = SqueezeAndBinarize(input, _g.BINARIZE_THRESHOLD)
    output = SqueezeAndBinarize(output, _g.BINARIZE_THRESHOLD)


    VisualizeVolxel(input)
    VisualizeVolxel(output)
    # PrintVoxel(input,'bathtub_te_1_input.csv')
    # PrintVoxel(output, 'bathtub_te_1_output.csv')

if __name__== '__main__':
    main(sys.argv)