import os, sys
import random

ROOTPATH='D:/Libs/Keras/VCONV_DAE/VCONV_DAE_TF/Data/KNU_DownSampled/'
DATAPATH = ROOTPATH + 'Data/'
EXTENSION = '.csv'
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

def TrainValidationSplit(fileList, shuffle):
    if shuffle == True:
        random.shuffle(fileList)

    count = int(len(fileList)*TRAINVAL_RATIO)
    return fileList[:count], fileList[count:]

def PrintText(fileList, outFilename):
    f = open(ROOTPATH+outFilename,'w')
    for file in fileList:
        str = '{0} {1}\n'.format(0, DATAPATH+file+EXTENSION)
        f.write(str)
    f.close()

def main(argv):
    fileList = ListUpFiles(DATAPATH,EXTENSION)
    trainList, valList = TrainValidationSplit(fileList, True)

    PrintText(trainList, 'train_files.txt')
    PrintText(valList, 'val_files.txt')

if __name__ == '__main__':
    main(sys.argv)