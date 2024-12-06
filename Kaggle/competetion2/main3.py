from itertools import product
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras._tf_keras.keras.callbacks import EarlyStopping, Callback,CallbackList
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.optimizers.legacy import SGD
from matplotlib import pyplot
import csv

def divValTrainSet(X,Y):
    ValSet = np.random.choice(X.shape[0],int(X.shape[0]*0.2),replace=False)
    TrainSet = np.delete(np.arange(0, Y.shape[0] ), ValSet)
    XVal = X[ValSet,:]
    YVal = Y[ValSet]
    X = X[TrainSet,:]
    Y = Y[TrainSet]
    return (XVal, YVal, X, Y)


# Hyperparameter-Suche
def build_and_train_model(input_dim, layers, activation, optimizer, epochs, patience):
    # Modell erstellen
    model = Sequential()
    model.add(Dense(layers[0], input_dim=input_dim, activation=activation))
    for units in layers[1:]:
        model.add(Dense(units, activation=activation, use_bias=False))
    model.add(Dense(1, activation='linear', use_bias=False))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

    # Callbacks definieren
    checkpoint = keras.callbacks.ModelCheckpoint(
        'best_model_temp.h5', monitor='val_mean_squared_error', verbose=False,
        save_weights_only=True, save_best_only=True
    )
    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_mean_squared_error', patience=patience, verbose=False,
        restore_best_weights=True, mode='min'
    )
    callbacks_list = [checkpoint, earlystop]

    # Training
    history = model.fit(
        XTr, YTr, epochs=epochs, validation_data=(XVal, YVal), callbacks=callbacks_list, verbose=False
    )

    # Lade die besten Gewichte
    model.load_weights('best_model_temp.h5')
    val_loss = min(history.history['val_mean_squared_error'])

    return model, val_loss

# Lade die Daten
xTrain = np.load("XTrain.npy")
xTest = np.load("XTest.npy")
yTrain = np.load("YTrain.npy")
yMin = yTrain.min(axis=0); yMax = yTrain.max(axis=0)
Y=((yTrain-yMin)/(yMax-yMin))
(XVal, YVal, XTr, YTr) = divValTrainSet(xTrain,Y)

# Definiere Hyperparameter-Räume
layer_configs = [
    [64, 32], [64, 32, 16], [128, 64, 32],[48,24,16]
]
activations = ['relu', 'sigmoid']
optimizers = ['adam']
epochs = 200
patience = 20

# Hyperparameter-Kombinationen testen
best_model = None
best_loss = float('inf')
best_config = None

for layers, activation, optimizer in product(layer_configs, activations, optimizers):
    print(f"Testing configuration: Layers={layers}, Activation={activation}, Optimizer={optimizer}")
    model, val_loss = build_and_train_model(XTr.shape[1], layers, activation, optimizer, epochs, patience)
    print(f"Validation Loss: {val_loss}")
    if val_loss < best_loss:
        best_loss = val_loss
        best_model = model
        best_config = (layers, activation, optimizer)

print(f"Beste Konfiguration: Layers={best_config[0]}, Activation={best_config[1]}, Optimizer={best_config[2]}, Loss={best_loss}")

# Vorhersagen mit dem besten Modell
yp = best_model.predict(xTest).reshape(-1)
ypo = yp * (yMax - yMin) + yMin

assert len(yp) == 5344, f"Es müssen 5344 Vorhersagen sein, aber es gibt {len(ypo)}."
# CSV-Datei erstellen
filename = 'yPredict_Best_Garb_Simon.csv'
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "predicted"])
    for idx, prediction in enumerate(ypo, start=1):
        writer.writerow([idx, prediction])

print(f"CSV-Datei '{filename}' erfolgreich erstellt!")
