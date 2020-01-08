# Global Parameter Setting

# Training-------------------------------------------------
# Network Param
INPUT_VOXEL_SIZE=[30,30,30]
AUGMENTED_DROPOUT_RATE=.5
CONV_NUM_FILTERS=[64, 256, 256, 64, 1]
CONV_FILTER_SIZES=[9,4, 5, 6]
CONV_STRIDES=[3,2, 2, 3]
DESC_DIMS=[6912,6912]
# Training Param
LEARNING_RATE=0.1
DECAY=1e-6
MOMENTUM=0.9
BATCH=32
NUM_CLASS=31
NUM_EPOCHS = 10
TRAIN_OUTPUT_FOLDER = 'CKPT/'
# Training Data Param
DATA_SOURCE_ROOT = 'Data/ModelNet30/'
TRAIN_FILE_PATH = DATA_SOURCE_ROOT + 'train_files.txt'
VAL_FILE_PATH = DATA_SOURCE_ROOT + 'val_files.txt'

# Prediction--------------------------------------------------
WEIGHT_FILE_NAME = 'Epoch_08.h5'
PREDICTION_FILE_PATH = 'Data/ModelNet30/chair_te/chair_te_1.csv'
BINARIZE_THRESHOLD = 0.5