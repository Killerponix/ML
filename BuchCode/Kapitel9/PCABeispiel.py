import numpy as np

np.random.seed(42)
xs = np.arange(0.2,0.8,0.0025)
ys = np.arange(0.2,0.8,0.0025)
zs = np.arange(0.2,0.8,0.0025)

dx = 0.15*(np.random.rand(xs.shape[0])-0.5)
dy = 0.30*(np.random.rand(ys.shape[0])-0.5)
dz = 0.20*(np.random.rand(zs.shape[0])-0.5)

x = 0.5*xs + 0.25*ys + 0.3*zs + dx
y = 0.3*xs + 0.45*ys + 0.3*zs + dy
z = 0.1*xs + 0.30*ys + 0.6*zs + dz
dataset = np.vstack( (x,y,z) ).T

xbar = np.mean(dataset,axis=0) 
sigma = np.std(dataset,axis=0)
X = (dataset - xbar) / sigma

Sigma = np.cov(X.T)
(lamb, w) = np.linalg.eig(Sigma)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(2)
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(x,y,z,c='red',s=60,alpha=0.3)
ax.set_xlim([0,1]); ax.set_ylim([0,1]); ax.set_zlim([0,1])
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
xM = np.array([xbar[0],xbar[0],xbar[0]])
yM = np.array([xbar[1],xbar[1],xbar[1]])
zM = np.array([xbar[2],xbar[2],xbar[2]])
D = np.zeros_like(w)
D[:,0] = lamb[0]/4*w[:,0]
D[:,1] = lamb[1]/4*w[:,1]
D[:,2] = lamb[2]/4*w[:,2]
ax.quiver(xM,yM,zM, D[0,:], D[1,:], D[2,:])

xNew = (w[:,0].T@X.T).T
print(np.std(xNew))