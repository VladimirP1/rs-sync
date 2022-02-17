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


#-----------------------

rng = default_rng()
sum = np.array([0.,0.,0.])
lmed = np.inf
for i in range(100):
  x = m[rng.choice(m.shape[0], size=3, replace=False)]
  v = np.cross(x[0,:] - x[1,:],x[0,:] - x[2,:])
  v = v/np.linalg.norm(v)

  residuals = m @ v
  med = np.median(residuals * residuals)
  if med < lmed:
    lmed = med
    li = i
    lv = v
    weights = np.sqrt((1 / (1 + (residuals * 1e3) ** 2))[np.newaxis].T)

print(li, lmed)
print(lv)

sol = lv

#def fun(x, p):
#  return 1e3 * p @ (x/ np.linalg.norm(x)) + 1e-9 * x[0] + 1e-9 * x[1] + 1e-9 * x[2]
#
#lsq = least_squares(fun, lv, args=(m,), loss='cauchy', max_nfev = 500, diff_step = 1e-9)
#
#sol = lsq['x']
#
#sol = sol / np.linalg.norm(sol)
#
#print('smooth sol error: ', np.arccos(sol.dot(lv)) * 180 / np.pi)
#
#print(sol)

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
#m = weights * m
ax.scatter(m[:,0], m[:,1], m[:,2], c=weights, cmap='jet')

fig.colorbar(plt.cm.ScalarMappable(cmap = 'jet'), ax = ax)

plt.show()


#------------------------

for i in range(1000):
    svd = np.linalg.svd(weights * m)
    sol = svd[2].T[:,-1]
    residuals = m @ sol
    weights = np.sqrt((1 / (1 + (residuals * 1e3) ** 2))[np.newaxis].T)

print('smooth sol error: ', np.arccos(sol.dot(lv)) * 180 / np.pi)

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
