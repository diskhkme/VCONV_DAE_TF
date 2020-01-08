import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import losses

import keras.backend as K

from DataGenerator import DataGenerator
from Model import build_model
import GlobalParam as _g

def ConstructModel():
    K.clear_session()

    model = build_model(input_voxel_size=_g.INPUT_VOXEL_SIZE,
                        augmenting_dropout_rate=_g.AUGMENTED_DROPOUT_RATE,
                        conv_num_filters=_g.CONV_NUM_FILTERS,
                        conv_filter_sizes=_g.CONV_FILTER_SIZES,
                        conv_strides=_g.CONV_STRIDES,
                        desc_dims=_g.DESC_DIMS)

    sgd = optimizers.SGD(lr=_g.LEARNING_RATE, decay=_g.DECAY, momentum=_g.MOMENTUM)

    model.compile(loss='binary_crossentropy',optimizer=sgd)

    return model

def GenerateDataset():
    train_generator = DataGenerator(_g.TRAIN_FILE_PATH,
                                   batch_size=_g.BATCH, dim=_g.INPUT_VOXEL_SIZE,
                                   n_classes=_g.NUM_CLASS, shuffle=True)
    val_generator = DataGenerator(_g.VAL_FILE_PATH,
                                   batch_size=_g.BATCH, dim=_g.INPUT_VOXEL_SIZE,
                                   n_classes=_g.NUM_CLASS, shuffle=True)

    print("Number of images in the training dataset:/t{:>6}".format(train_generator.get_dataset_siez()))

    return train_generator, val_generator

def Train(model, train_generator, val_generator):
    model_checkpoint = ModelCheckpoint(filepath=_g.TRAIN_OUTPUT_FOLDER + 'Epoch_{epoch:02d}.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
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
                        use_multiprocessing=True,
                        workers=6,
                        epochs=_g.NUM_EPOCHS,
                        callbacks=callbacks,
                        validation_data=val_generator)

def main(argv):
    model = ConstructModel()
    print(model.summary())

    trainGenerator,val_generator = GenerateDataset()

    print(model.get_layer('output').output_shape)
    Train(model,trainGenerator,val_generator)

if __name__== '__main__':
    main(sys.argv)