import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.optimize import least_squares

with open('problem.xyz', 'r') as f:
  m = []
  for line in f.readlines():
    xx = line.strip().replace('  ', ' ').split(' ')
    m.append([*map(float,xx)])
m = np.array(m)


weights = np.ones((m.shape[0],1))
sol = np.array([[1,0,0]])
for i in range(1000):
    svd = np.linalg.svd(weights * m)
    sol = svd[2].T[:,-1]
    residuals = m @ sol
    weights = np.sqrt((1 / (1 + (residuals * 1e3) ** 2))[np.newaxis].T)

#print('smooth sol error: ', np.arccos(sol.dot(lv)) * 180 / np.pi)

#-----------------------


fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(projection='3d')
ax.view_init(-10,90)

normal = sol
xx, yy = np.meshgrid(np.linspace(-.02,.02,50), np.linspace(-.02,.02,50))
zz = (-normal[0] * xx - normal[1] * yy) * 1. / normal[2]
xx = xx[np.abs(zz) < .02]
yy = yy[np.abs(zz) < .02]
zz = zz[np.abs(zz) < .02]
ax.scatter(xx, yy, zz, alpha=0.2)
ax.scatter(m[:,0], m[:,1], m[:,2], c=weights, cmap='jet')

fig.colorbar(plt.cm.ScalarMappable(cmap = 'jet'), ax = ax)

plt.show()
