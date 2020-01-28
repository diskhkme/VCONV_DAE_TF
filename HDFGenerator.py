import os, sys
import numpy as np
from scipy.ndimage import zoom
import h5py
import random

ROOTPATH='D:/KHK/Lib/VCONV_DAE_TF/Data/ModelNet30_CVT30/Data/'
OUTPATH = 'D:/KHK/Lib/VCONV_DAE_TF/Data/ModelNet30_CVT64_New/'
EXTENSION = '.csv'
TRAINOUTFILENAME = 'train_dataset.h5'
VALOUTFILENAME = 'val_dataset.h5'
INSIZE = [30,30,30]
OUTSIZE = [64,64,64]
TRAINVAL_RATIO = 0.8

def ListUpFiles(path,ext):
    files = os.listdir(path)
    fileList = []

    for file in files:
        stext = os.path.splitext(file)
        assert(stext.count != 2)
        fileName = stext[0]
        extension = stext[1]
        if ext == extension:
            fileList.append(fileName)

    return fileList

def createHDF5Dataset(fileList):

    random.shuffle(fileList)
    count = int(len(fileList)*TRAINVAL_RATIO)

    trainFileList =fileList[:count]
    valFileList =fileList[count:]

    WriteHDF(trainFileList, TRAINOUTFILENAME)
    WriteHDF(valFileList,VALOUTFILENAME)

def WriteHDF(fileList, outFile):
    hdf5Dataset = h5py.File(OUTPATH + '/' + outFile, 'w')
    hdf5Voxel = hdf5Dataset.create_dataset(name='Voxel', shape=(len(fileList),),
                                           dtype=h5py.special_dtype(vlen=np.uint8))

    for i, file in enumerate(fileList):
        csv_data = np.loadtxt(ROOTPATH + file + EXTENSION, delimiter=',')
        csv_data = csv_data.astype(int)
        rData = np.reshape(csv_data, INSIZE)
        if (rData.shape[0] != OUTSIZE[0]):
            rData = zoom(rData, (OUTSIZE[0]/INSIZE[0],OUTSIZE[1]/INSIZE[1],OUTSIZE[2]/INSIZE[2]), order=0)
        assert(rData.shape[0] == OUTSIZE[0])
        rData = rData.reshape(-1)
        # rData = rData.astype(int)

        hdf5Voxel[i] = rData
        if i % 100 == 0:
            print('H5 writing {0} / {1}'.format(i, len(fileList)))

    hdf5Dataset.close()

def KNULoadTxt(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    data = lines[2].split(',')
    data = list(map(int, data))
    data = np.array(data)

    return data

def main(argv):
    fileList = ListUpFiles(ROOTPATH,EXTENSION)
    createHDF5Dataset(fileList)

if __name__ == '__main__':
    main(sys.argv)