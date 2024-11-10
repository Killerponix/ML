import numpy as np
import pandas as pd
from CARTRegressionTree import bRegressionTree

f = open("hourCleanUp.csv")
header = f.readline().rstrip('\n')  # skip the header
featureNames = header.split(',')
dataset = np.loadtxt(f, delimiter=",")
dataset = dataset[dataset[:, 8] != 4]
split = dataset[:, 3]  # Extrahiere die 'day'-Spalte
dataset = np.delete(dataset, 3, axis=1)  # Entferne die 'day'-Spalte
# Filtere nach `weathersit`

f.close()

X = dataset[:, 0:12]
Y = dataset[:, 14]

#X = np.delete(X,6, axis=1)

index = np.flatnonzero(X[:, 8] == 4)
X = np.delete(X, index, axis=0)
Y = np.delete(Y, index, axis=0)
np.random.seed(42)

train_indices = np.flatnonzero(split <= 20)  # Trainingsdaten bis Day 20
test_indices = np.flatnonzero(split > 20)    # Testdaten ab Day 21

# Teile X und Y in Trainings- und Testdaten auf
XTrain = X[train_indices, :]
yTrain = Y[train_indices]
XTest = X[test_indices, :]
yTest = Y[test_indices]


# MainSet = np.arange(0, X.shape[0])
# Trainingsset = X[split <= 20]  #Bis Day 20
# Testset = X[split > 20]  #Ab Day 21
# #Trainingsset = np.random.choice(X.shape[0], int(0.8*X.shape[0]), replace=False)
# #Testset = np.delete(MainSet, Trainingsset)
# XTrain = X[Trainingsset, :]
# yTrain = Y[Trainingsset]
# XTest = X[Testset, :]
# yTest = Y[Testset]

myTree = bRegressionTree(minLeafNodeSize=15, threshold=2)
myTree.fit(XTrain, yTrain)
yPredict = np.round(myTree.predict(XTest))
import matplotlib.pyplot as plt

plt.figure(1)
yDiff = yPredict - yTest
plt.hist(yDiff, 22, color='gray')
plt.xlim(-200, 200)
plt.title('Fehler auf Testdaten')
plt.figure(2)
plt.hist(yTest, 22, color='gray')
plt.title('Testdaten')
plt.show()
print('Mittlere Abweichung: %e ' % (np.mean(np.abs(yDiff))))
