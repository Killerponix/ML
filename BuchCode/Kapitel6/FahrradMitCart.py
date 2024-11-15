import numpy as np
from CARTRegressionTree import bRegressionTree

f = open("hourCleanUp.csv")
header = f.readline().rstrip('\n')  # skip the header
featureNames = header.split(',')
dataset = np.loadtxt(f, delimiter=",")
f.close()

X = dataset[:,0:13]
Y = dataset[:,15]

#X = np.delete(X,6, axis=1)

index = np.flatnonzero(X[:,8]==4)
X = np.delete(X,index, axis=0)
Y = np.delete(Y,index, axis=0)

np.random.seed(42)
MainSet = np.arange(0,X.shape[0])
Trainingsset = np.random.choice(X.shape[0], int(0.8*X.shape[0]), replace=False)
Testset = np.delete(MainSet,Trainingsset)
XTrain = X[Trainingsset,:]
yTrain = Y[Trainingsset]
XTest = X[Testset,:]
yTest = Y[Testset]

myTree = bRegressionTree(minLeafNodeSize=15,threshold=2)
myTree.fit(XTrain,yTrain)
yPredict = np.round(myTree.predict(XTest))
import matplotlib.pyplot as plt
plt.figure(1)
yDiff = yPredict - yTest
plt.hist(yDiff,22,color='gray')
plt.xlim(-200,200)
plt.title('Fehler auf Testdaten')
plt.figure(2)
plt.hist(yTest,22,color='gray')
plt.title('Testdaten')
print('Mittlere Abweichung: %e ' % (np.mean(np.abs(yDiff))))
