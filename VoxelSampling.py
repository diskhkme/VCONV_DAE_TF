import os, sys
import numpy as np
from scipy.ndimage import zoom

ROOTPATH='D:/KHK/Lib/VCONV_DAE_TF/Data/ModelNet30_CVT64/Data/'
OUTPATH = 'D:/KHK/Lib/VCONV_DAE_TF/Data/ModelNet30_CVT128/Data/'
EXTENSION = '.csv'
OUTEXTENSION = '.csv'
INSIZE = [64,64,64]
OUTSIZE = [128,128,128]

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

def Downsample(fileList):
    for file in fileList:
        # csv_data = KNULoadTxt(ROOTPATH+file+EXTENSION)
        csv_data = np.loadtxt(ROOTPATH + file + EXTENSION, delimiter=',')
        data = np.reshape(csv_data, INSIZE)
        rData = zoom(data, (OUTSIZE[0]/INSIZE[0],OUTSIZE[1]/INSIZE[1],OUTSIZE[2]/INSIZE[2]))
        assert(rData.shape[0] == OUTSIZE[0])
        rData = np.reshape(rData, (OUTSIZE[0],OUTSIZE[1]*OUTSIZE[2]))
        rData.astype(int)
        np.savetxt(OUTPATH+file+OUTEXTENSION, rData, fmt='%d', delimiter=',')
        print(file)


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
    Downsample(fileList)

if __name__ == '__main__':
    main(sys.argv)