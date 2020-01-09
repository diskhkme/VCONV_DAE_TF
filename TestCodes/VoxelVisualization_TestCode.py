from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import csv


file = 'D:/Test_Models/VAE/Data/1.txt'
f = open(file,'r')
lines = f.readlines()
f.close()

data = lines[2].split(',')
data = list(map(int,data))
data = np.array(data)

reshapedData = np.reshape(data, [128,128,128])

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_aspect('equal')

ax.voxels(reshapedData, edgecolor="k")

plt.show()