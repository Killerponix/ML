import numpy as np
from sklearn import svm

fFloat  = open("Auto2MerkmaleClass.csv","r")
dataset = np.loadtxt(fFloat, delimiter=",")
fFloat.close()
yTrain = dataset[:,0]
x = dataset[:,1:3] 
xMin = x.min(axis=0); xMax = x.max(axis=0) 
XTrain = (x - xMin) / (xMax - xMin)

svmRBF = svm.SVC(kernel='rbf', decision_function_shape='ovr', gamma=1)
svmRBF.fit(XTrain,yTrain) 

XX, YY = np.mgrid[0:1:0.01, 0:1:0.01]
X = np.array([XX.ravel(), YY.ravel()]).T
yP = svmRBF.predict(X)

indexA = np.flatnonzero(yTrain==0)
indexB = np.flatnonzero(yTrain==1)
indexC = np.flatnonzero(yTrain==2)
indexD = np.flatnonzero(yTrain==3)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
yP = yP.reshape(XX.shape)
ax.pcolormesh(XX, YY, yP, cmap=plt.cm.tab20)
ax.scatter(XTrain[indexA,0],XTrain[indexA,1],color='black', marker='o')
ax.scatter(XTrain[indexB,0],XTrain[indexB,1],color='black', marker='x')
ax.scatter(XTrain[indexC,0],XTrain[indexC,1],color='black', marker='+')
ax.scatter(XTrain[indexD,0],XTrain[indexD,1],color='black', marker='*')
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_title("Klassifikation mit RBF ($\gamma=1$)")

