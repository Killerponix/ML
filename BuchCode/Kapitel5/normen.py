import numpy as np
import matplotlib.pyplot as plt

def plotUnitCircle(p, sampels):
    x = 3*np.random.rand(sampels,2)-1.5
    n = np.linalg.norm(x,p,axis=1)
    indexIn = np.flatnonzero(n <= 1)
    indexOut = np.flatnonzero(n > 1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x[indexOut,0],x[indexOut,1],c='red',s=60,alpha=0.1, marker='*')
    ax.scatter(x[indexIn,0],x[indexIn,1],c='black',s=60, marker='+')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True,linestyle='-',color='0.75')
    
plotUnitCircle(np.inf, 5000)