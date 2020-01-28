import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import losses
import keras.backend as K
from keras.utils.multi_gpu_utils import multi_gpu_model

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
import h5py

from DataGenerator import DataGenerator
from Model import build_model_30, build_model_64, build_model_128
import GlobalParam as _g

import pyvox.models
import pyvox.writer

def PrintVoxel(data, filepath):
    # data = data.reshape((_g.INPUT_VOXEL_SIZE[0],_g.INPUT_VOXEL_SIZE[1]* _g.INPUT_VOXEL_SIZE[2]))
    # data.astype(int)
    # np.savetxt(filepath, data, delimiter=',', fmt='%d')

    data = data.reshape((_g.INPUT_VOXEL_SIZE[0],_g.INPUT_VOXEL_SIZE[1], _g.INPUT_VOXEL_SIZE[2]))
    data.astype(int)
    vox = pyvox.models.Vox.from_dense(data)
    pyvox.writer.VoxWriter(filepath,vox).write()


def VisualizeVolxel(data):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')

    ax.voxels(data, edgecolor="k")

    plt.show()

def SqueezeAndBinarize(data, threshold):
    if(len(data.shape) > 3):
        if data.shape[0] == 1:
            data = np.squeeze(data, axis=0)
        data = np.squeeze(data, axis=3)

    # cv2.imshow('',data[:,:,5])
    data = data > threshold

    return data

def ConstructModel():
    K.clear_session()

    if _g.TYPE == '30':
        # ModelNet30
        model = build_model_30(input_voxel_size=_g.INPUT_VOXEL_SIZE,
                            augmenting_dropout_rate=_g.AUGMENTED_DROPOUT_RATE,
                            conv_num_filters=_g.CONV_NUM_FILTERS,
                            conv_filter_sizes=_g.CONV_FILTER_SIZES,
                            conv_strides=_g.CONV_STRIDES,
                            desc_dims=_g.DESC_DIMS)
    elif _g.TYPE == '64':
        # KNU_Simplification
        model = build_model_64(input_voxel_size=_g.INPUT_VOXEL_SIZE,
                            augmenting_dropout_rate=_g.AUGMENTED_DROPOUT_RATE,
                            conv_num_filters=_g.CONV_NUM_FILTERS,
                            conv_filter_sizes=_g.CONV_FILTER_SIZES,
                            conv_strides=_g.CONV_STRIDES,
                            desc_dims=_g.DESC_DIMS)
    elif _g.TYPE == '128':
        # KNU_Simplification
        model = build_model_128(input_voxel_size=_g.INPUT_VOXEL_SIZE,
                                   augmenting_dropout_rate=_g.AUGMENTED_DROPOUT_RATE,
                                   conv_num_filters=_g.CONV_NUM_FILTERS,
                                   conv_filter_sizes=_g.CONV_FILTER_SIZES,
                                   conv_strides=_g.CONV_STRIDES,
                                   desc_dims=_g.DESC_DIMS)

    sgd = optimizers.SGD(lr=_g.LEARNING_RATE, decay=_g.DECAY, momentum=_g.MOMENTUM)

    model = multi_gpu_model(model, gpus=2)
    model.compile(loss='binary_crossentropy',optimizer=sgd)

    model.load_weights(_g.TRAIN_OUTPUT_FOLDER + _g.WEIGHT_FILE_NAME)

    return model

def MakePrediction(model, input):
    output = model.predict(input, batch_size=2)

    return output

def LoadInput(filename):
    #csv_data = np.loadtxt('Data/ModelNet30/bathtub_te/bathtub_te_1.csv'delimiter=',')

    inputData = np.empty((len(filename), _g.INPUT_VOXEL_SIZE[0], _g.INPUT_VOXEL_SIZE[1],_g.INPUT_VOXEL_SIZE[2], 1),dtype=int)

    ind = 0
    for file in filename:
        csv_data = np.loadtxt(file, delimiter = ',')
        data = np.reshape(csv_data, (_g.INPUT_VOXEL_SIZE[0], _g.INPUT_VOXEL_SIZE[1],_g.INPUT_VOXEL_SIZE[2]))
        data = np.expand_dims(data, axis=3)
        inputData[ind,:,:,:] = data
        ind = ind+1

    return inputData

def LoadFromHDF(load_HDF_path, inds):
    assert(os.path.exists(load_HDF_path))

    HDF_data = h5py.File(load_HDF_path, 'r')

    inputData = np.empty((len(inds), _g.INPUT_VOXEL_SIZE[0], _g.INPUT_VOXEL_SIZE[1], _g.INPUT_VOXEL_SIZE[2], 1),
                         dtype=int)
    ind = 0
    for input in HDF_data['Voxel'][inds]:
        input = np.reshape(input, (_g.INPUT_VOXEL_SIZE[0], _g.INPUT_VOXEL_SIZE[1], _g.INPUT_VOXEL_SIZE[2]))
        input = np.expand_dims(input, axis=3)
        inputData[ind,:,:,:] = input
        ind = ind + 1

    return inputData

def main(argv):
    if _g.FROM_HDF:
        input = LoadFromHDF(_g.VAL_HDF_FILE_PATH, _g.PREDICTION_DATA_INDEX)
    else:
        input = LoadInput(_g.PREDICTION_FILE_PATH)

    model = ConstructModel()
    print(model.summary())

    output = MakePrediction(model, input)

    for i in range(input.shape[0]):
        inputSingle = SqueezeAndBinarize(input[i,], _g.BINARIZE_THRESHOLD)
        outputSingle = SqueezeAndBinarize(output[i,], _g.BINARIZE_THRESHOLD)

        # VisualizeVolxel(inputSingle)
        # VisualizeVolxel(outputSingle)
        PrintVoxel(inputSingle, _g.TRAIN_OUTPUT_FOLDER + "{0}_Input.vox".format(i))
        PrintVoxel(outputSingle, _g.TRAIN_OUTPUT_FOLDER + "{0}_Output.vox".format(i))



if __name__== '__main__':
    main(sys.argv)