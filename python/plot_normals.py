#!/usr/bin/python3

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

plt.rcParams.update({
    "pgf.texsystem": "pdflatex"
})

frame = 600

dirs = np.genfromtxt('dump/motion.csv', delimiter=',')
m = np.genfromtxt('dump/normals_{}.csv'.format(frame), delimiter=',')
normal = dirs[dirs[:,0] == frame][0, 1:]


fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(projection='3d')
ax.view_init(-14,-24)

ax.scatter(m[:,0], m[:,1], m[:,2], c = np.abs(np.dot(m, normal)), cmap='jet')

# bounding box
minv,maxv = np.min(m), np.max(m)
rad = (maxv - minv) / 2
center = ((np.min(m[:,0]) + np.max(m[:,0])) / 2, (np.min(m[:,1]) + np.max(m[:,1])) / 2, (np.min(m[:,2]) + np.max(m[:,2])) / 2)
ax.set_xlim((center[0] - rad , center[0] + rad))
ax.set_ylim((center[1] - rad , center[1] + rad))
ax.set_zlim((center[2] - rad , center[2] + rad))

#
axis = np.cross((0,0,1), normal)
angle = np.arcsin(np.linalg.norm(axis)) + np.pi/2
rot = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle)

#
circle = np.array([np.array((np.sin(x), np.cos(x), 0)) for x in np.linspace(0,2*np.pi,100)])
circle = rot.apply(circle) * rad
circle += center - np.dot(center, normal) * normal

verts = [list(zip(circle[:,0], circle[:,1], circle[:,2]))]
srf = Poly3DCollection(verts, alpha=.25, facecolor='orange')
plt.gca().add_collection3d(srf)

qlen = rad / 2
ax.quiver([0],[0],[0],[qlen],[0],[0], colors='r')
ax.quiver([0],[0],[0],[0],[qlen],[0], colors='g')
ax.quiver([0],[0],[0],[0],[0],[qlen], colors='b')

# ax.scatter(m[:,0], m[:,1], m[:,2], c=weights, cmap='jet')
# fig.colorbar(plt.cm.ScalarMappable(cmap = 'jet'), ax = ax)

fig.tight_layout()

plt.show()

print('azim =', ax.azim, ', elev =', ax.elev)

plt.savefig('fig.pgf', backend='pgf')