import numpy as np

dataset = np.loadtxt("iris.csv", delimiter=",")
x = dataset[:,0:4]
y = dataset[:,4]
percentTrainingset = 0.8
np.random.seed(42)
TrainSet     = np.random.choice(x.shape[0],int(x.shape[0]*percentTrainingset),replace=False)
XTrain       = x[TrainSet,:]
yTrain       = y[TrainSet]
TestSet      = np.delete(np.arange(0,len(y)), TrainSet) 
XTest        = x[TestSet,:]
yTest        = y[TestSet]

from tensorflow.keras.utils import to_categorical
YTrain = to_categorical(yTrain-1, 3)
YTest  = to_categorical(yTest-1, 3)

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
ANN = Sequential()
ANN.add(layers.Dense(8,activation='tanh',input_dim=4))
ANN.add(layers.Dense(8,activation='tanh'))
ANN.add(layers.Dense(3,activation='softmax'))
ANN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
ANN.fit(XTrain,YTrain,epochs=500, verbose=False)
scores = ANN.evaluate(XTest,YTest)
print("Accuracy: %.2f%%" % (scores[1]*100))