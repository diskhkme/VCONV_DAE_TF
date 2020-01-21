# Global Parameter Setting

# Training-------------------------------------------------
# Network Param (30 size, paper)
# TYPE='30' # 30 / 64 / 128
# INPUT_VOXEL_SIZE=[30,30,30]
# AUGMENTED_DROPOUT_RATE=.5
# CONV_NUM_FILTERS=[64, 256, 256, 64, 1]
# CONV_FILTER_SIZES=[9, 4, 5, 6]
# CONV_STRIDES=[3,2,2,3]
# DESC_DIMS=[6912,6912]

# Network Param (64 size)
TYPE='64' # 30 / 64 / 128
INPUT_VOXEL_SIZE=[64,64,64]
AUGMENTED_DROPOUT_RATE=.5
CONV_NUM_FILTERS=[64, 128, 256, 256, 128, 64, 1]
CONV_FILTER_SIZES=[9, 6, 3, 3, 5, 4]
CONV_STRIDES=[3,2,2,2,2,3]
DESC_DIMS=[16384,16384] # first conv layer has "same" padding

# Network Param (128 size)
# INPUT_VOXEL_SIZE=[128,128,128]
# AUGMENTED_DROPOUT_RATE=.5
# CONV_NUM_FILTERS=[16, 64, 64, 64, 64, 16, 1]
# CONV_FILTER_SIZES=[12,8,8, 8,8, 17]
# CONV_STRIDES=[3,2,2, 2,2, 3]
# DESC_DIMS=[8000,8000]



# Training Param
LEARNING_RATE=0.1
DECAY=1e-6
MOMENTUM=0.9
BATCH=32
NUM_CLASS=31
NUM_EPOCHS = 30
TRAIN_OUTPUT_FOLDER = 'CKPT/KNU_64/'
# Training Data Param
DATA_SOURCE_ROOT = 'Data/KNU_64/'
TRAIN_FILE_PATH = DATA_SOURCE_ROOT + 'train_files.txt'
VAL_FILE_PATH = DATA_SOURCE_ROOT + 'val_files.txt'

# Prediction--------------------------------------------------
WEIGHT_FILE_NAME = 'Epoch_06_0.008840.h5'
PREDICTION_FILE_PATH = 'Data/KNU_64/Data/1.csv'
BINARIZE_THRESHOLD = 0.5