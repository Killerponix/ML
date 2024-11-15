import numpy as np 
import matplotlib.pyplot as plt

XX, YY = np.mgrid[0:1:0.01, 0:1:0.01]
X = np.array([XX.ravel(), YY.ravel()]).T
Z = np.sin(XX**2)**2 + np.log(1 + YY**2)
fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.pcolormesh(XX, YY, Z, cmap=plt.cm.Set1)
ax = fig.add_subplot(1,2,2)
ax.contourf(XX, YY, Z, cmap=plt.cm.Set1)