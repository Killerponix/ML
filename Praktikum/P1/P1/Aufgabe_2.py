import numpy as np
import sys
sys.path.append("C:/Users/Simon/Desktop/Studium/5.Semester/Maschinelles Lernen/ML/Programme/Model")
from Programme.Model.CARTDecisionTree import *
from Programme.Model.util.accuracy import *
import matplotlib.pyplot as plt


train = np.loadtxt(delimiter=",",fname="Trainingsset.csv")
test = np.loadtxt(delimiter=",",fname="Testset.csv")
data = np.loadtxt(delimiter=",",fname="AllData.csv")

y_train = train[:, 0]       # Erste Spalte ist y für das Trainingsset

X_train1 = train[:,7]
X_train2 = train[:,10]
X_train = (X_train1+X_train2).reshape(-1,1)

y_test = test[:, 0]         # Erste Spalte ist y für das Testset

X_test1 = test[:,7]
X_test2 = test[:,10]
X_test = (X_test1+X_test2).reshape(-1,1)

CC = bDecisionTree(xDecimals=5,threshold=0.1,minLeafNodeSize=3)
CC.fit(X_train,y_train)
y_predict = CC.predict(X_test)
print(y_predict - y_test)
accuracy(y_test,y_predict)

XX,YY = np.mgrid[X_train.min():X_train.max():0.005, X_train.min():X_train.max():0.005]
X= np.array([XX.ravel(),YY.ravel()]).T
Z=np.sin(XX**2)**2+np.log(1+YY**2)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.pcolormesh(XX,YY,Z,cmap=plt.cm.Set1)
ax.scatter(X_test,X_test)
# ax = fig.add_subplot(1,2,2)
# ax.contourf(XX,YY,Z, cmap=plt.cm.Set1)
fig.show(True)