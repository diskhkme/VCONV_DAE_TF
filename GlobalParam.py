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
CONV_NUM_FILTERS=[16, 32, 1024, 1024, 32, 16, 1]
CONV_FILTER_SIZES=[7, 5, 3, 3, 7, 10]
CONV_STRIDES=[3,3,2,2,3,3]
DESC_DIMS=[8192,8192]

# Network Param (64 size)
# TYPE='64deeper' # 30 / 64 / 128
# INPUT_VOXEL_SIZE=[64,64,64]
# AUGMENTED_DROPOUT_RATE=.5
# CONV_NUM_FILTERS=[64, 128, 256, 512, 512, 256, 128, 64, 1]
# CONV_FILTER_SIZES=[4, 3, 3, 3, 3, 3, 3, 4]
# CONV_STRIDES=[2,2,2,2,2,2,2,2,2]
# DESC_DIMS=[13824,13824]

# Network Param (128 size)
# TYPE='128'
# INPUT_VOXEL_SIZE=[128,128,128]
# AUGMENTED_DROPOUT_RATE=.5
# CONV_NUM_FILTERS=[64, 128, 256, 256, 128, 64, 1]
# CONV_FILTER_SIZES=[9, 4, 3,  3, 5, 16]
# CONV_STRIDES=[4,3,3,  3, 3, 4]
# DESC_DIMS=[6912,6912]


# Data Generator Param
LOAD_ON_MEMORY = True
LOAD_WEIGHT = True
LOAD_WEIGHT_PATH = 'CKPT/ModelNet30_CVT64_New/Epoch_03_0.122206.h5'


# Training Param
# Mode
# 'KNU_Simplification' : KNU txt 파일을 그대로 사용하는 경우
# 'KNU_CSV_Converted' : KNU txt 파일을 csv로 변환하여 사용하는 경우
# 'ModelNet30' : ModelNet30 csv 파일 사용하는 경우

MODE = 'KNU_Simplification'
LEARNING_RATE=0.1
DECAY=1e-6
MOMENTUM=0.9
BATCH=64
NUM_CLASS=31
NUM_EPOCHS = 30
TRAIN_OUTPUT_FOLDER = 'CKPT/ModelNet30_CVT64/'
# Training Data Param
DATA_SOURCE_ROOT = 'Data/ModelNet30_CVT64/'
TRAIN_FILE_PATH = DATA_SOURCE_ROOT + 'train_files.txt'
VAL_FILE_PATH = DATA_SOURCE_ROOT + 'val_files.txt'

# Prediction--------------------------------------------------
WEIGHT_FILE_NAME = 'Epoch_95_0.180470.h5'
PREDICTION_FILE_PATH = ['Data/KNU_64/Data/disk.csv','Data/KNU_64/Data/20A.csv','Data/KNU_64/Data/cab.csv','Data/KNU_64/Data/gear_1.csv']
# PREDICTION_FILE_PATH = ['Data/KNU_64/Data/disk.csv','Data/KNU_64/Data/20A.csv','Data/KNU_64/Data/cab.csv','Data/KNU_64/Data/gear_1.csv']
BINARIZE_THRESHOLD = 0.5