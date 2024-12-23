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
np.random.seed(7)
# tf.random.set_seed(42)  # Optional: Seed für TensorFlow setzen

# Lade die Daten
xTrain = np.load("XTrain.npy")
xTest = np.load("XTest.npy")
yTrain = np.load("YTrain.npy")
yMin = yTrain.min(axis=0); yMax = yTrain.max(axis=0)
Y=((yTrain-yMin)/(yMax-yMin))

def divValTrainSet(X,Y):
    ValSet = np.random.choice(X.shape[0],int(X.shape[0]*0.18),replace=False)
    TrainSet = np.delete(np.arange(0, Y.shape[0] ), ValSet)
    XVal = X[ValSet,:]
    YVal = Y[ValSet]
    X = X[TrainSet,:]
    Y = Y[TrainSet]
    return (XVal, YVal, X, Y)




def build_train(model,units,activ):
    myANN=model
    myANN.add(Dense(units[0],input_dim=12,activation=activ[0]))
    myANN.add(Dense(units[1],activation=activ[1],use_bias=False))
    myANN.add(Dense(units[2],activation=activ[2],use_bias=False))
    myANN.add(Dense(units[3],activation=activ[3],use_bias=False))
    myANN.add(Dense(1,activation='linear',use_bias=False))
    myANN.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
    myANN.save('StartANN.h5')
    checkpoint = keras.callbacks.ModelCheckpoint('bestW.weights.h5', monitor='val_mean_squared_error', verbose=False,save_weights_only=True, save_best_only=True)
    earlystop = keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=20, verbose=False,restore_best_weights=True,mode='min')
    callbacksList = [checkpoint,earlystop]
    history=myANN.fit(XTr,YTr,epochs=200,validation_data=(XVal, YVal),callbacks=callbacksList)

    # myANN.load_weights('bestW.weights.h5')
    # myANN = load_model('StartAnn.h5')
    # myANN.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
    # history=myANN.fit(XTr,YTr,epochs=200,validation_data=(XVal, YVal),callbacks=callbacksList)
    # myANN.load_weights('bestW.weights.h5')
    # myANN = load_model('StartAnn.h5')
    # myANN.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
    # history=myANN.fit(XTr,YTr,epochs=200,validation_data=(XVal, YVal),callbacks=callbacksList)
    val_loss = min(history.history['val_mean_squared_error'])

    return myANN, val_loss


(XVal, YVal, XTr, YTr) = divValTrainSet(xTrain,Y)
# bestmse = 1000000
# confunits =[[128,64,32,16],[96,48,32,24],[128,64,32,24],[48,32,24,16]]
# conf_act = [['relu','relu','relu','relu'],['relu','relu','sigmoid','relu'],['relu','sigmoid','elu','relu'],['elu','elu','elu','elu'],['relu','relu','sigmoid','sigmoid']]
# best_units=0
# best_act =0
# for units in confunits:
#     for activ in conf_act:
#         myANN = Sequential()
#         # myANN = load_model('StartAnn.h5')
#         print(f"Testing: layers={units}, Activation={activ}")
#         myANN, mse = build_train(myANN,units,activ)
#         print(f"Validation Loss: {mse}")
#         if(mse<bestmse):
#             bestmse=mse
#             best_act=activ
#             best_units=units
#             myANN.save_weights('bestmse2.weights.h5')
#             best_conf = (best_units,best_act)

myANN = Sequential()
# myANN = load_model('StartAnn.h5')
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
history=myANN.fit(XTr,YTr,epochs=600,validation_data=(XVal, YVal),callbacks=callbacksList)

# myANN.load_weights('bestW.weights.h5')
# myANN = load_model('StartAnn.h5')
# myANN.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
# history=myANN.fit(XTr,YTr,epochs=300,validation_data=(XVal, YVal),callbacks=callbacksList)
# myANN.load_weights('bestW.weights.h5')
# myANN = load_model('StartAnn.h5')
# myANN.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
# history=myANN.fit(XTr,YTr,epochs=300,validation_data=(XVal, YVal),callbacks=callbacksList)
# val_loss = min(history.history['val_mean_squared_error'])
# print(best_conf)
myANN.load_weights('bestW.weights.h5')
# myANN.load_weights('Garb_Simon.weights.h5')
yp = myANN.predict(xTest)
yp = yp.reshape(yp.shape[0])


# errorT = (yMax - yMin)*(yp - yTest)
# print(np.mean(np.abs(errorT)))
# myANN = load_model('StartANN.h5')
# myANN.compile(loss='mean_squared_error', optimizer='SGD')

# earlystop = keras.callbacks.EarlyStopping(monitor='val_loss ', patience=20, verbose=False,restore_best_weights=True)
# callbacksList = [earlystop,checkpoint]
# # myANN.load_weights('bestW.weights.h5')
# history = myANN.fit(XTr,YTr, epochs=10, validation_data=(XVal, YVal), callbacks=
# callbacksList, verbose=False)
# import matplotlib.pyplot as plt
# lossMonitor = np.array(history.history['loss'])
# valLossMonitor = np.array(history.history['val_loss'])
# counts = np.arange(lossMonitor.shape[0])
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.plot(counts,lossMonitor,'k', label='Trainingsdaten')
# ax.plot(counts,valLossMonitor,'r:', label='Validierungsdaten')
# ax.set_xlabel('Lernzyklus')
# ax.set_ylabel('Fehler')
# ax.legend()
# myANN = load_model('StartANN.h5')
# checkpoint = keras.callbacks.ModelCheckpoint('bestW.h5', monitor='val_loss', verbose=False,save_weights_only=True, save_best_only=True)
# callbacksList = [checkpoint,earlystop]
# myANN.load_weights('bestW.weights.h5')
# myANN.compile(loss='mean_squared_error', optimizer='SGD')
# history = myANN.fit(XTr,YTr, epochs=10, validation_data=(XVal, YVal), callbacks=callbacksList, verbose=False)
# myANN.load_weights('bestW.weights.h5')
# myANN.compile(loss='mean_squared_error', optimizer='SGD')
# np.argmin(valLossMonitor)
# myANN.layers[i].set_weights(listOfNumpyArrays)
# listOfNumpyArrays = myANN.layers[i].get_weights()

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
