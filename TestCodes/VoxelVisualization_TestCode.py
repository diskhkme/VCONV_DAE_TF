from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
N1 = 10
N2 = 10
N3 = 10
ma = np.random.choice([0,1], size=(N1,N2,N3), p=[0.99, 0.01])

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_aspect('equal')

ax.voxels(ma, edgecolor="k")

plt.show()