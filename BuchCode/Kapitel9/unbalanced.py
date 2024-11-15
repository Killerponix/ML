import numpy as np
import matplotlib.pyplot as plt
from CARTDecisionTree import bDecisionTree
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

dataset = np.loadtxt("iris.csv", delimiter=",")
Xall = dataset[:,0:4]
Yall = dataset[:,4] -1
X = Xall[0:125,:]
Y = Yall[0:125]

np.random.seed(42)
n = X.shape[0]
MainSet = np.arange(0,n)
Trainingsset = np.random.choice(n, int(0.75*n), replace=False)
Testset = np.delete(MainSet,Trainingsset)
Testset = np.delete(MainSet,Trainingsset)
XTrain = X[Trainingsset,:]
yTrain = Y[Trainingsset] 
XTest = np.vstack( (X[Testset,:],Xall[125:150,:] ))
yTest = np.hstack( (Y[Testset], Yall[125:150]) ) 

noOfClasses = 3
YTrain = to_categorical(yTrain, noOfClasses)
YTest  = to_categorical(yTest, noOfClasses)

ANN = Sequential()
ANN.add(layers.Dense(8,activation='tanh',input_dim=4))
ANN.add(layers.Dense(3,activation='softmax'))
ANN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
w = np.ones(XTrain.shape[0])
w[yTrain==2] = 2
ANN.fit(XTrain,YTrain,epochs=500, verbose=True, sample_weight=w)
yPred  = ANN.predict(XTest)
choise = np.argmax(yPred, axis=1)
errors = np.abs(yTest -  choise).sum()/yTest.shape[0]
print("Accuracy: %.2f%%" % ((1-errors)*100))

konfusionMatrix  = np.zeros((noOfClasses,noOfClasses))
konfusionMatrixN = np.zeros((noOfClasses,noOfClasses))
for i in range(noOfClasses):
    index = np.flatnonzero(yTest == i)
    for j in range(noOfClasses):
        index2 = np.flatnonzero(choise[index] == j)
        konfusionMatrix[i,j] = len(index2)
        konfusionMatrixN[i,j] = len(index2)/len(index)
print(konfusionMatrix)
print(konfusionMatrixN)