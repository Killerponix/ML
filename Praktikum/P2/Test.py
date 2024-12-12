import numpy as np
import pandas as pd
import sklearn.datasets
import keras
from keras import layers, regularizers
from keras.models import Sequential
from keras.callbacks import EarlyStopping

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

def div_val_train_set(X, Y):
    """
    Teilt die Daten in Trainings-, Validierungs- und Testdatensätze auf (75% train, 15% test, 15% validierung).
    """
    total_indices = np.arange(X.shape[0])
    np.random.shuffle(total_indices)

    val_size = int(X.shape[0] * 0.15)
    test_size = int(X.shape[0] * 0.15)

    val_indices = total_indices[:val_size]
    test_indices = total_indices[val_size:val_size + test_size]
    train_indices = total_indices[val_size + test_size:]

    X_val = X[val_indices, :]
    Y_val = Y[val_indices]
    X_train = X[train_indices, :]
    Y_train = Y[train_indices]
    X_test = X[test_indices, :]
    Y_test = Y[test_indices]

    return X_val, Y_val, X_train, Y_train, X_test, Y_test

np.random.seed(42)
X, Y = sklearn.datasets.load_breast_cancer(return_X_y=True)

# Datensätze teilen
X_val, Y_val, X_train, Y_train, X_test, Y_test = div_val_train_set(X, Y)

# Normalisierung der Trainingsdaten
X_train_norm, train_min_vals, train_max_vals = min_max_normalize(X_train)

# Normalisierung der Validierungs- und Testdaten basierend auf Trainingsdaten
X_val_norm, _, _ = min_max_normalize(X_val, train_min_vals, train_max_vals)
X_test_norm, _, _ = min_max_normalize(X_test, train_min_vals, train_max_vals)

# Modell-Definition
myANN = Sequential([
    layers.Dense(64, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dense(32, activation='sigmoid', use_bias=True, kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dense(16, activation='sigmoid', use_bias=True, kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dense(1, activation='sigmoid', use_bias=True, kernel_regularizer=regularizers.l2(0.0001))
])

# Modell kompilieren
opt = keras.optimizers.Adam(learning_rate=0.1)
myANN.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=500, verbose=True, restore_best_weights=True, mode='min')
callbacks_list = [early_stop]

# Training
history = myANN.fit(X_train_norm, Y_train, epochs=2000, validation_data=(X_val_norm, Y_val), callbacks=callbacks_list, verbose=0)

# Vorhersagen und Evaluation
y_pred = (myANN.predict(X_test_norm) > 0.5).astype(int)

# Genauigkeit berechnen
accuracy = np.mean(Y_test == y_pred.flatten())
print("Accuracy:", accuracy)
