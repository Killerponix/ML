import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras._tf_keras.keras.callbacks import EarlyStopping, Callback,CallbackList
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.optimizers.legacy import SGD
from matplotlib import pyplot

xTrain = np.load("XTrain.npy")
xTest = np.load("XTest.npy")
yTrain = np.load("YTrain.npy")
yMin = yTrain.min(axis=0); yMax = yTrain.max(axis=0)
Y=((yTrain-yMin)/(yMax-yMin))

# ANN = load_model('StartAnn.h5')
units =[[128,64,32,16],[96,48,32,24],[128,64,32,24],[48,32,24,16]]
activ = [['relu','relu','relu','relu'],['relu','relu','sigmoid','relu'],['relu','sigmoid','elu','relu'],['elu','elu','elu','elu'],['relu','relu','sigmoid','sigmoid']]
myANN = Sequential()
myANN.add(Dense(units[2][0],input_dim=12,activation=activ[0][0]))
myANN.add(Dense(units[2][1],activation=activ[0][1],use_bias=False))
myANN.add(Dense(units[2][2],activation=activ[0][2],use_bias=False))
myANN.add(Dense(units[2][3],activation=activ[0][3],use_bias=False))
myANN.add(Dense(1,activation='linear',use_bias=False))


# for layer in ANN.layers:
#     print(layer.name, layer.input_shape, layer.output_shape)



myANN.load_weights('bestmse2.weights.h5')
yp = myANN.predict(xTest)
ypo = yp * (yMax - yMin) + yMin

import csv
assert len(ypo) == 5344, f"Es müssen 5344 Vorhersagen sein, aber es gibt {len(yp)}."
# Speichere die Vorhersagen in einer CSV-Datei
filename = 'yPredict_zw_Garb_Simon.csv'
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "predicted"])
    for idx, prediction in enumerate(yp, start=1):
        writer.writerow([idx, prediction])

print(f"CSV-Datei '{filename}' erfolgreich erstellt!")
