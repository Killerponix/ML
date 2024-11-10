import numpy as np
import matplotlib.pyplot as plt


data = np.array([[1, 0,0],
                [0.9, 0.5,0],
                [0.5, 0.9,0],
                [0, 1,0],
                [-0.5, 0.9,0],
                [-0.8, 0.5,0],
                [-1, 0,0],
                [0, 0.5,1],
                [0.1, 0,1],
                [0.5, -0.4,1],
                [1, -0.5,1],
                [1.5, -0.4,1],
                [1.9, 0,1],
                [2, 0.5,1]]
                )
plt.figure()
plt.scatter(data[:,0],data[:,1],c=data[:,2])
plt.show()

