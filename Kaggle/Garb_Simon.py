import numpy as np
import tensorflow as tf
from tensorflow import keras
# from keras._tf_keras.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from keras._tf_keras.keras.callbacks import EarlyStopping, Callback,CallbackList
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.optimizers.legacy import SGD
from matplotlib import pyplot
# tf.keras.optimizers.legacy.SGD(learning_rate=0.1)


import csv

# Eager Execution aktivieren (optional)
tf.config.experimental_run_functions_eagerly(True)

# Funktion zur Min-Max-Normalisierung
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

# Seed für Reproduzierbarkeit
np.random.seed(42)
# tf.random.set_seed(42)  # Optional: Seed für TensorFlow setzen

# Lade die Daten
xTrain = np.load("XTrain.npy")
xTest = np.load("XTest.npy")
yTrain = np.load("YTrain.npy")
yMin = yTrain.min(axis=0); yMax = yTrain.max(axis=0)
Y=((yTrain-yMin)/(yMax-yMin))

def divValTrainSet(X,Y):
    ValSet = np.random.choice(X.shape[0],int(X.shape[0]*0.2),replace=False)
    TrainSet = np.delete(np.arange(0, Y.shape[0] ), ValSet)
    XVal = X[ValSet,:]
    YVal = Y[ValSet]
    X = X[TrainSet,:]
    Y = Y[TrainSet]
    return (XVal, YVal, X, Y)







(XVal, YVal, XTr, YTr) = divValTrainSet(xTrain,Y)


myANN = Sequential()
myANN.add(Dense(64,input_dim=12,activation='sigmoid'))
myANN.add(Dense(32,activation='sigmoid',use_bias=False))
myANN.add(Dense(16,activation='sigmoid',use_bias=False))
myANN.add(Dense(8,activation='sigmoid',use_bias=False))
myANN.add(Dense(1,activation='linear',use_bias=False))
myANN.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
myANN.save('StartANN.h5')
checkpoint = keras.callbacks.ModelCheckpoint('bestW.weights.h5', monitor='val_mean_squared_error', verbose=False,save_weights_only=True, save_best_only=True)
earlystop = keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=50, verbose=False,restore_best_weights=True,mode='min')
callbacksList = [checkpoint,earlystop]
history=myANN.fit(XTr,YTr,epochs=500,validation_data=(XVal, YVal),callbacks=callbacksList)

myANN.load_weights('bestW.weights.h5')
yp = myANN.predict(xTest)
yp = yp.reshape(yp.shape[0])
ypo = yp * (yMax - yMin) + yMin

assert len(ypo) == 5344, f"Es müssen 5344 Vorhersagen sein, aber es gibt {len(ypo)}."
# Speichere die Vorhersagen in einer CSV-Datei
filename = 'yPredict2_Garb_Simon.csv'
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "predicted"])
    for idx, prediction in enumerate(ypo, start=1):
        writer.writerow([idx, prediction])

print(f"CSV-Datei '{filename}' erfolgreich erstellt!")
