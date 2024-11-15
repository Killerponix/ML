import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf 
np.random.seed(42)
tf.random.set_seed(42) 


X = np.loadtxt("BostonFeature.csv", delimiter=",")
Y = np.loadtxt("BostonTarget.csv", delimiter=",")

yMin = Y.min(axis=0); yMax = Y.max(axis=0) 
Y = (Y - yMin) / (yMax - yMin) 
TrainSet     = np.random.choice(X.shape[0],int(X.shape[0]*0.80), replace=False)
XTrain       = X[TrainSet,:] 
YTrain       = Y[TrainSet]
TestSet      = np.delete(np.arange(0, len(Y) ), TrainSet) 
XTest        = X[TestSet,:]
YTest        = Y[TestSet]

myANN = Sequential()
myANN.add(Dense(10,input_dim=13,kernel_initializer='normal',activation='sigmoid'))
myANN.add(Dense(10,kernel_initializer='random_uniform',activation='sigmoid',use_bias=False))
myANN.add(Dense(1,kernel_initializer='normal',activation='linear',use_bias=False))
myANN.compile(loss='mean_squared_error', optimizer='adam')
myANN.save('StartANN.h5')
history = myANN.fit(XTrain,YTrain, epochs=1000, verbose=False)

yp = myANN.predict(XTest)
yp = yp.reshape(yp.shape[0])
errorT = (yMax - yMin)*(yp - YTest)
print(np.mean(np.abs(errorT)))

from tensorflow.keras.models import load_model
myANN = load_model('StartANN.h5')

def divValTrainSet(X,Y):
    ValSet    = np.random.choice(X.shape[0],int(X.shape[0]*0.25),replace=False)
    TrainSet  = np.delete(np.arange(0, Y.shape[0] ), ValSet) 
    XVal     = X[ValSet,:]
    YVal     = Y[ValSet]
    X        = X[TrainSet,:]
    Y        = Y[TrainSet]
    return (XVal, YVal, X, Y)
(XVal, YVal, XTr, YTr) = divValTrainSet(XTrain,YTrain)

from tensorflow import keras
earlystop  = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=False,  restore_best_weights=True)
callbacksList = [earlystop] 

history = myANN.fit(XTr,YTr, epochs=1000, validation_data=(XVal, YVal), callbacks=callbacksList, verbose=False)

import matplotlib.pyplot as plt
lossMonitor = np.array(history.history['loss'])
valLossMonitor = np.array(history.history['val_loss'])
counts = np.arange(lossMonitor.shape[0])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(counts,lossMonitor,'k', label='Trainingsdaten')
ax.plot(counts,valLossMonitor,'r:', label='Validierungsdaten')
ax.set_xlabel('Lernzyklus')
ax.set_ylabel('Fehler')
ax.legend()

yp = myANN.predict(XTrain)
yp = yp.reshape(yp.shape[0])
error = (yMax - yMin)*(yp - YTrain)
print(np.mean(np.abs(error)))

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.set_title('Verteilung der Abweichungen auf der Trainingsmenge')
ax.hist(error,color='gray')
ax.set_xlabel('Abweichung in Tausenden')
ax.set_ylabel('Anzahl')
ax = fig.add_subplot(1,2,2)
ax.set_title('Verteilung der Abweichungen auf der Testmenge')
ax.hist(errorT,color='gray')
ax.set_xlabel('Abweichung in Tausenden')
ax.set_ylabel('Anzahl')

myANN = load_model('StartANN.h5')
checkpoint = keras.callbacks.ModelCheckpoint('bestW.h5', monitor='val_loss', verbose=False, save_weights_only=True, save_best_only=True)
callbacksList = [checkpoint] 
history = myANN.fit(XTr,YTr, epochs=1000, validation_data=(XVal, YVal), callbacks=callbacksList, verbose=False)
myANN.load_weights('bestW.h5')
yp = myANN.predict(XTest)
yp = yp.reshape(yp.shape[0])
errorT = (yMax - yMin)*(yp - YTest)
print(np.mean(np.abs(errorT)))

lossMonitor = np.array(history.history['loss'])
valLossMonitor = np.array(history.history['val_loss'])
counts = np.arange(lossMonitor.shape[0])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(counts,lossMonitor,'k', label='Trainingsdaten')
ax.plot(counts,valLossMonitor,'r:', label='Validierungsdaten')
ax.set_xlabel('Lernzyklus')
ax.set_ylabel('Fehler')
ax.legend()
