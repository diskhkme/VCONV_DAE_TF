import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import losses

import keras.backend as K

from DataGenerator import DataGenerator
from Model import build_model

def ConstructModel():
    K.clear_session()

    model = build_model(input_voxel_size=[30,30,30],
                        augmenting_dropout_rate=.5,
                        conv_num_filters=[64, 256, 256, 64, 1],
                        conv_filter_sizes=[9,4, 5, 6],
                        conv_strides=[3,2, 2, 3],
                        desc_dims=[6912,6912])

    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9)

    model.compile(loss='binary_crossentropy',optimizer=sgd)

    return model

def GenerateDataset():
    train_generator = DataGenerator('Data/ModelNet30/train_files.txt',
                                   batch_size=32, dim=(30, 30, 30),
                                   n_classes=31, shuffle=True)
    val_generator = DataGenerator('Data/ModelNet30/val_files.txt',
                                   batch_size=32, dim=(30, 30, 30),
                                   n_classes=31, shuffle=True)

    print("Number of images in the training dataset:/t{:>6}".format(train_generator.get_dataset_siez()))

    return train_generator, val_generator

def Train(model, train_generator, val_generator):
    model_checkpoint = ModelCheckpoint(filepath='CKPT/' + 'Epoch_{epoch:02d}.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto',
                                       period=1)

    csv_logger = CSVLogger(filename='CKPT/' + 'VAE_Training_Log.csv',
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
                        epochs=10,
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