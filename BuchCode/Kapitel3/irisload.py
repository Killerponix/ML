import numpy as np

fString = open("iris.data","r")
fFloat  = open("iris.csv","w")

for line in fString:
    line = line.replace("Iris-setosa", "1")
    line = line.replace("Iris-versicolor", "2")
    line = line.replace("Iris-virginica", "3")
    fFloat.write(line)
    
fString.close()
fFloat.close()

fFloat  = open("iris.csv","r")
#header = fFloat.readline().rstrip('\n')  # skip the header
#ColumnNames = header.split(',')
dataset = np.loadtxt(fFloat, delimiter=",")
fFloat.close()

import matplotlib.pyplot as plt

fig = plt.figure(1)

ax = fig.add_subplot(2,2,1)
ax.scatter(dataset[0:50,0],dataset[0:50,1],c='red',s=60,alpha=0.6)
ax.scatter(dataset[50:100,0],dataset[50:100,1],c='green',marker='^',s=60,alpha=0.6)
ax.scatter(dataset[100:150,0],dataset[100:150,1],c='blue',marker='*',s=80,alpha=0.6)
ax.set_xlabel('Kelchblattlaenge (cm)')
ax.set_ylabel('Kelchblattbreite (cm)')
ax.grid(True,linestyle='-',color='0.75')

ax = fig.add_subplot(2,2,2)
ax.scatter(dataset[0:50,2],dataset[0:50,3],c='red',s=60,alpha=0.6)
ax.scatter(dataset[50:100,2],dataset[50:100,3],c='green',marker='^',s=60,alpha=0.6)
ax.scatter(dataset[100:150,2],dataset[100:150,3],c='blue',marker='*',s=80,alpha=0.6)
ax.set_xlabel('Kronblattlaenge (cm)')
ax.set_ylabel('Kronblattbreite (cm)')
ax.grid(True,linestyle='-',color='0.75')

ax = fig.add_subplot(2,2,3)
ax.scatter(dataset[0:50,0],dataset[0:50,2],c='red',s=60,alpha=0.6)
ax.scatter(dataset[50:100,0],dataset[50:100,2],c='green',marker='^',s=60,alpha=0.6)
ax.scatter(dataset[100:150,0],dataset[100:150,2],c='blue',marker='*',s=80,alpha=0.6)
ax.set_xlabel('Kelchblattlaenge (cm)')
ax.set_ylabel('Kronblattlaenge (cm)')
ax.grid(True,linestyle='-',color='0.75')

ax = fig.add_subplot(2,2,4)
ax.scatter(dataset[0:50,1],dataset[0:50,3],c='red',s=60,alpha=0.6)
ax.scatter(dataset[50:100,1],dataset[50:100,3],c='green',marker='^',s=60,alpha=0.6)
ax.scatter(dataset[100:150,1],dataset[100:150,3],c='blue',marker='*',s=80,alpha=0.6)
ax.set_xlabel('Kelchblattbreite (cm)')
ax.set_ylabel('Kronblattbreite (cm)')
ax.grid(True,linestyle='-',color='0.75')

plt.tight_layout()
plt.show(block=False)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(2)
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(dataset[0:50,1],dataset[0:50,2],dataset[0:50,3],c='red',s=60,alpha=0.6)
ax.scatter(dataset[50:100,1],dataset[50:100,2],dataset[50:100,3],c='green',marker='^',s=60,alpha=0.6)
ax.scatter(dataset[100:150,1],dataset[100:150,2],dataset[100:150,3],c='blue',marker='*',s=80,alpha=0.6)
ax.set_xlabel('Kelchblattbreite (cm)')
ax.set_ylabel('Kronblattlaenge (cm)')
ax.set_zlabel('Kronblattbreite (cm)')
plt.show()
