import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sys import argv



fname = argv[1]
xs, ys, zs = [], [], []
idx = 0
with open(fname, 'r') as file:
    for line in file:
        if line[0] == '#':
            continue
        x, y, z, w, p, q, r= [float(x) for x in line.split()]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        
        # Rotation on origin stays the same
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_zlim(0, 2)
        fig.savefig("{}.png".format(idx))
        plt.close(fig)
        idx += 1

