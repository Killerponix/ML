import numpy as np
import pandas as pd
import sklearn.datasets
import keras
import tensorflow as tf
from keras import layers
from keras import regularizers
from keras._tf_keras.keras.models import Sequential
# Sequential = from keras.models import Sequential
from keras._tf_keras.keras.callbacks import EarlyStopping, Callback,CallbackList
from keras._tf_keras.keras.models import load_model
from keras.src.layers import Dense
from Programme.Model.util.accuracy import accuracy


def min_max_normalize(data, min_vals=None, max_vals=None):
    """
    Normalisiert die Daten mit Min-Max-Normalisierung.
    """
    if min_vals is None:
        min_vals = data.min(axis=0)  # Minimum pro Spalte
    if max_vals is None:
        max_vals = data.max(axis=0)  # Maximum pro Spalte
    normalized = (data - min_vals) / (max_vals - min_vals)
    return normalized, min_vals, max_vals

def divValTrainSet(X,Y):
    ValSet = np.random.choice(X.shape[0],int(X.shape[0]*0.15),replace=False)
    TrainSet = np.delete(np.arange(0, Y.shape[0] ), ValSet)
    testSet = np.random.choice(TrainSet.shape[0],int(TrainSet.shape[0]*0.15),replace=False)
    TrainSet = np.delete(np.arange(0, Y.shape[0] ), testSet)
    XVal = X[ValSet,:]
    YVal = Y[ValSet]
    Xtrain = X[TrainSet,:]
    Ytrain = Y[TrainSet]
    Xtest = X[testSet,:]
    Ytest = [testSet]
    return (XVal, YVal, Xtrain, Ytrain,Xtest,Ytest)


X,Y = sklearn.datasets.load_breast_cancer(return_X_y=True)

Xnorm= min_max_normalize(X) #Normalisieren nur auf die Trainingsmenge, die Daten muss man dann nacher nochmal
# normalisieren, also die test und validierungsmenge -- Muss hier noch geändert werden
# X = data[:, 2:]
# Y=data[:, 1:2]
print(Y)
Xval, Yval, xtrain, ytrain, xtest,ytest = divValTrainSet(X,Y)


myANN = Sequential()
myANN.add(Dense(64,activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
myANN.add(Dense(32,activation='sigmoid',use_bias=True,kernel_regularizer=regularizers.l2(0.0001)))
myANN.add(Dense(16,activation='sigmoid',use_bias=True,kernel_regularizer=regularizers.l2(0.0001)))
myANN.add(Dense(8,activation='sigmoid',use_bias=True,kernel_regularizer=regularizers.l2(0.0001)))
myANN.add(Dense(2,activation='sigmoid',use_bias=True,kernel_regularizer=regularizers.l2(0.0001)))
opt = keras.optimizers.Adam(learning_rate=0.01)
myANN.compile(loss='binary_crossentropy', optimizer=opt, metrics=[keras.metrics.Accuracy()])

# checkpoint = keras.callbacks.ModelCheckpoint('bestW.weights.h5', monitor='accuarcy', verbose=False,
#                                              save_weights_only=True, save_best_only=True)

earlystop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, verbose=False,
                                          restore_best_weights=True,mode='max')
callbacksList = [earlystop]
history=myANN.fit(xtrain,ytrain,epochs=30,validation_data=(Xval, Yval)) #callbacks=callbacksList

y_pred = myANN.predict(xtest)
accuracy(ytest,y_pred)
