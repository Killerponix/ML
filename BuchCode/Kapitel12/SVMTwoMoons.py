import numpy as np
from twoMoonsProblem import twoMoonsProblem
from sklearn import svm

(XTrain,yTrain) = twoMoonsProblem()

svmP2 = svm.SVC(kernel='poly', degree=7, decision_function_shape='ovr', C=1000)
svmP2.fit(XTrain,yTrain) 

XX, YY = np.mgrid[-1:2:0.01, -1:2:0.01]
X = np.array([XX.ravel(), YY.ravel()]).T
yP = svmP2.predict(X)

indexA = np.flatnonzero(yTrain>0.5)
indexB = np.flatnonzero(yTrain<0.5)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
yP = yP.reshape(XX.shape)
ax.pcolormesh(XX, YY, yP, cmap=plt.cm.Set1)
ax.scatter(XTrain[indexA,0],XTrain[indexA,1],color='white', marker='o')
ax.scatter(XTrain[indexB,0],XTrain[indexB,1],color='black', marker='+')
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_xlim([-1,2])
ax.set_ylim([-1,2])
ax.set_title("Klassifikation mit Polynom-Kernel (degree=7, C=1000)")


svmRBF = svm.SVC(kernel='rbf', decision_function_shape='ovr')
svmRBF.fit(XTrain,yTrain) 
yP = svmRBF.predict(X)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
yP = yP.reshape(XX.shape)
ax.pcolormesh(XX, YY, yP, cmap=plt.cm.Set1)
ax.scatter(XTrain[indexA,0],XTrain[indexA,1],color='white', marker='o')
ax.scatter(XTrain[indexB,0],XTrain[indexB,1],color='black', marker='+')
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_xlim([-1,2])
ax.set_ylim([-1,2])
ax.set_title("Klassifikation mit RBF-Kernel")

svmLinear = svm.SVC(kernel='linear', decision_function_shape='ovr')
svmLinear.fit(XTrain,yTrain) 
yP = svmLinear.predict(X)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
yP = yP.reshape(XX.shape)
ax.pcolormesh(XX, YY, yP, cmap=plt.cm.Set1)
ax.scatter(XTrain[indexA,0],XTrain[indexA,1],color='white', marker='o')
ax.scatter(XTrain[indexB,0],XTrain[indexB,1],color='black', marker='+')
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_xlim([-1,2])
ax.set_ylim([-1,2])
ax.set_title("Klassifikation mit linearem Kernel")

svmSIG = svm.SVC(kernel='sigmoid', decision_function_shape='ovr')
svmSIG.fit(XTrain,yTrain) 
yP = svmSIG.predict(X)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
yP = yP.reshape(XX.shape)
ax.pcolormesh(XX, YY, yP, cmap=plt.cm.Set1)
ax.scatter(XTrain[indexA,0],XTrain[indexA,1],color='white', marker='o')
ax.scatter(XTrain[indexB,0],XTrain[indexB,1],color='black', marker='+')
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_xlim([-1,2])
ax.set_ylim([-1,2])
ax.set_title("Klassifikation mit Sigmoid-Kernel")