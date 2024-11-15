import numpy as np
from CARTRegressionTree import bRegressionTree 

np.random.seed(42)
x = 10*np.random.rand(1000,2)
y = np.zeros(1000)
index = np.flatnonzero(x[:,0]<2)
y[index] = 1
index = np.flatnonzero(np.logical_and(x[:,0] >= 2,x[:,1]<5))
y[index] = 1

MainSet = np.arange(0,1000)
Trainingsset = np.random.choice(1000, 800, replace=False)
Testset = np.delete(MainSet,Trainingsset)
XTrain = x[Trainingsset,:]
yTrain = y[Trainingsset]
XTest = x[Testset,:]
yTest = y[Testset]

smallTree = bRegressionTree()
smallTree.fit(XTrain,yTrain)

noise = 0.1*np.random.rand(1000) - 0.05
y = y + noise
yTrain = y[Trainingsset]
yTest = y[Testset]

complexTree = bRegressionTree()
complexTree.fit(XTrain,yTrain)

yPredict = complexTree.predict(XTest)
error = np.abs(yPredict - yTest)
print(error.mean())
yPredict = smallTree.predict(XTest)
error = np.abs(yPredict - yTest)
print(error.mean())

ValSet = np.random.choice(800, 200, replace=False)
xVal = XTrain[ValSet]
yVal = yTrain[ValSet]
Trainingsset = np.delete(Trainingsset,ValSet)
XTrain = x[Trainingsset,:]
yTrain = y[Trainingsset]

preTree = bRegressionTree(threshold = 2.5*10**-1)
preTree.fit(XTrain,yTrain)
yPredict = preTree.predict(xVal)
error = np.abs(yPredict - yVal)
print(error.mean())
