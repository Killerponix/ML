import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras._tf_keras.keras.callbacks import EarlyStopping, Callback,CallbackList
from keras._tf_keras.keras.models import load_model

from Programme.Model.util.TrainTestSplit import train_test_split
from Programme.Model.util.mse import mean_squared_error
# myDevice = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_visible_devices(devices= myDevice, device_type='GPU')
tf.config.experimental_run_functions_eagerly(True)

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


np.random.seed(42)
# tf.random.set(42)


xTrain = np.load("XTrain.npy")
xTest = np.load("XTest.npy")
yTrain = np.load("YTrain.npy")
print(xTrain.shape)

x_Train_N, min_vals, max_vals = min_max_normalize(xTrain)
x_Test_N, _, _ = min_max_normalize(xTest, min_vals=min_vals, max_vals=max_vals)

X_train, X_test, y_train, y_test = train_test_split(x_Train_N, yTrain, test_size=0.2)

myAnn = Sequential()
myAnn.add(Dense(10, input_dim=12, kernel_initializer='normal', activation='sigmoid'))
myAnn.add(Dense(10, kernel_initializer='random_uniform', activation='sigmoid', use_bias=False))
myAnn.add(Dense(1, kernel_initializer='normal', activation='linear', use_bias=False))
myAnn.compile(loss='mean_squared_error', optimizer='adam')
myAnn.save('StartAnn.h5')
history = myAnn.fit(X_train, y_train, epochs=10, verbose=False)
y_pred = myAnn.predict(X_test)
y_pred = y_pred.reshape(y_pred.shape[0])
myAnn = load_model('StartAnn.h5')

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=False,restore_best_weights=True)

# checkpoint = keras.callbacks.ModelCheckpoint('bestW.h5', monitor='val_loss', verbose=False,
#                                              save_weights_only=True, save_best_only=True)

callbacksList = [earlystop]
history = myAnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=
callbacksList, verbose=False)
myAnn = load_model('StartAnn.h5')
checkpoint = keras.callbacks.ModelCheckpoint('bestW.h5', monitor='val_loss', verbose=False,save_weights_only=True, save_best_only=True)
callbacksList = [checkpoint,earlystop]
history = myAnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=callbacksList, verbose=False)

myAnn.load_weights('bestW.h5')

# Vorhersage mit den besten Gewichten auf Testdaten
y_pred_best = myAnn.predict(X_test)




import csv
filename = 'yPredict_Garb_Simon.csv'
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "predicted"])
    for idx, prediction in enumerate(y_pred_best, start=1):
        writer.writerow([idx, prediction])

print(f"CSV-Datei '{filename}' erfolgreich erstellt!")
