import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from shutil import copyfile

from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import losses
from keras.utils.multi_gpu_utils import multi_gpu_model

import keras.backend as K

from DataGenerator import DataGenerator
from Model import build_model_30, build_model_64, build_model_128, build_model_64_deeper
import GlobalParam as _g

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
    elif _g.TYPE == '64deeper':
        # KNU_Simplification
        model = build_model_64_deeper(input_voxel_size=_g.INPUT_VOXEL_SIZE,
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

    if _g.LOAD_WEIGHT:
        model.load_weights(_g.LOAD_WEIGHT_PATH)

    return model

def GenerateDataset():
    train_generator = DataGenerator(
                                   batch_size=_g.BATCH, dim=_g.INPUT_VOXEL_SIZE,
                                   n_classes=_g.NUM_CLASS, shuffle=True,
                                   mode='KNU_CSV_Converted',
                                   load_data_into_memory = _g.LOAD_ON_MEMORY,
                                   load_HDF_path = _g.DATA_SOURCE_ROOT + 'train_dataset.h5')
    val_generator = DataGenerator(
                                   batch_size=_g.BATCH, dim=_g.INPUT_VOXEL_SIZE,
                                   n_classes=_g.NUM_CLASS, shuffle=True,
                                   mode='KNU_CSV_Converted',
                                   load_data_into_memory=_g.LOAD_ON_MEMORY,
                                   load_HDF_path=_g.DATA_SOURCE_ROOT + 'val_dataset.h5')

    print("Number of images in the training dataset:/t{:>6}".format(train_generator.get_dataset_siez()))

    return train_generator, val_generator

def Train(model, train_generator, val_generator):
    model_checkpoint = ModelCheckpoint(filepath=_g.TRAIN_OUTPUT_FOLDER + 'Epoch_{epoch:02d}_{val_loss:08f}.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='auto',
                                       period=1)

    csv_logger = CSVLogger(filename=_g.TRAIN_OUTPUT_FOLDER + 'VAE_Training_Log.csv',
                           separator=',',
                           append=True)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0.0,
                                   patience=5,
                                   verbose=1)

    callbacks = [model_checkpoint, csv_logger, early_stopping]

    model.fit_generator(generator=train_generator,
                        use_multiprocessing=False,
                        workers=4,
                        epochs=_g.NUM_EPOCHS,
                        callbacks=callbacks,
                        validation_data=val_generator)

def main(argv):
    copyfile('GlobalParam.py', _g.TRAIN_OUTPUT_FOLDER + 'GlobalParam.py')

    model = ConstructModel()
    print(model.summary())

    trainGenerator,val_generator = GenerateDataset()

    print(model.get_layer('output').output_shape)
    Train(model,trainGenerator,val_generator)



if __name__== '__main__':
    main(sys.argv)