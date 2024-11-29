import numpy as np
import sys
sys.path.append("C:/Users/Simon/Desktop/Studium/5.Semester/Maschinelles Lernen/ML/Programme/Model")
from Programme.Model.CARTDecisionTree import *
from Programme.Model.util.accuracy import *
import matplotlib.pyplot as plt

feature_ids = [0, 7, 10]
trainset = np.loadtxt("Trainingsset.csv", delimiter=',')
trainset = trainset[:, feature_ids]
x_train = trainset[:, 1:]
y_train = trainset[:, 0]
testset = np.loadtxt("Testset.csv", delimiter=',')
testset = testset[:, feature_ids]
x_test = testset[:, 1:]
y_test = testset[:, 0]


CC = bDecisionTree(xDecimals=5,threshold=0.1,minLeafNodeSize=3)
CC.fit(x_train,y_train)
y_predict = CC.predict(x_test)
print(y_predict - y_test)
accuracy(y_test,y_predict)
error_count = np.count_nonzero(y_test - y_predict)
errors = (y_test != y_predict)



XX, YY = np.mgrid[x_train[:, 0].min():x_train[:, 0].max():0.005, x_train[:, 1].min():x_train[:, 1].max():0.005]
X= np.array([XX.ravel(),YY.ravel()]).T
Z = CC.predict(X).reshape(XX.shape)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.pcolormesh(XX,YY,Z,cmap=plt.cm.plasma)
ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.prism)
ax.scatter(x_test[errors, 0], x_test[errors, 1], c="yellow", marker='x', s=100)
ax.set_xlabel("Flavanoids")
ax.set_ylabel("Color-Intensity")
fig.savefig("Aufgabe2.png", dpi=300)

plt.show()