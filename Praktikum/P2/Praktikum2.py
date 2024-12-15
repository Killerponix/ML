import gradio as gr
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from Programme.Model.util.mse import mean_squared_error
import numpy as np
import pandas as pd
import sklearn.datasets
import keras
from keras import layers, regularizers
from Programme.Model.util.accuracy import accuracy
from keras.callbacks import EarlyStopping
import os

np.random.seed(42)

# Funktion zum Erstellen des Modells
def create_model(neurons_per_layer, layers, activation_function, input_dim,l2rate,learning,opimizer):
    print(input_dim, neurons_per_layer, activation_function)
    # Eingabedimension überprüfen
    if not isinstance(input_dim, int):
        raise ValueError(f"Invalid input_dim: {input_dim}. Must be an integer.")

    model = Sequential()

    # Definieren der Eingabeform explizit
    # model.add(Dense(neurons_per_layer[0], activation=activation_function, input_dim=input_dim))
    model.add(Dense(neurons_per_layer, activation="sigmoid", input_dim=input_dim,use_bias=True,kernel_regularizer=regularizers.l2(l2rate)))
    neurons = neurons_per_layer
    # Hinzufügen der versteckten Schichten
    for i in range(layers):
        neurons = neurons // 2
        model.add(Dense(neurons, activation=activation_function,kernel_regularizer=regularizers.l2(l2rate),use_bias=True))

    # Ausgabe-Schicht
    model.add(Dense(1, activation='sigmoid',use_bias=True,kernel_regularizer=regularizers.l2(l2rate)))

    # Zusammenfassung und Kompilierung
    model.summary()
    if(opimizer == 'Adam'):
        opt = keras.optimizers.Adam(learning_rate=learning)
    elif(opimizer == 'SGD'):
        opt = keras.optimizers.SGD(learning_rate=learning)
    elif(opimizer== 'RMSprop'):
        opt = keras.optimizers.RMSprop(learning_rate=learning)

    # myANN.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    # model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
    return model


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


def div_val_train_set(X, Y,size):
    """
    Teilt die Daten in Trainings-, Validierungs- und Testdatensätze auf (75% train, 15% test, 15% validierung).
    """
    total_indices = np.arange(X.shape[0])
    np.random.shuffle(total_indices)

    val_size = int(X.shape[0] *size)
    test_size = int(X.shape[0] *size)

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


# Training des Modells
def train_model(neurons_per_layer, layers, activation_function, train_split, epochs, save_model, data_file,learn,l2rate,opt):

    if data_file is not None:
        # Datei verarbeiten und Daten laden
        print("TEST")
        data = filetransform(data_file)
        # X, Y = data[:,:-1], data[:,-1:]
        X, Y = data[:, :-1], data[:, -1]
        # xtest, ytest, xtrain, ytrain = divTestTrainSet(X, Y, train_split)
        X_val, Y_val, X_train, Y_train, X_test, Y_test = div_val_train_set(X, Y, train_split)
    else:
        X, Y = sklearn.datasets.load_breast_cancer(return_X_y=True)

    # Datensätze teilen
    X_val, Y_val, X_train, Y_train, X_test, Y_test = div_val_train_set(X, Y,train_split)

    xtmp, tmpmin,tmpmax = min_max_normalize(X)
    # Normalisierung der Trainingsdaten
    X_train_norm, train_min_vals, train_max_vals = min_max_normalize(X_train)

    # Normalisierung der Validierungs- und Testdaten basierend auf Trainingsdaten
    X_val_norm, _, _ = min_max_normalize(X_val, train_min_vals, train_max_vals)
    X_test_norm, _, _ = min_max_normalize(X_test, tmpmin, tmpmax)
    # X_val_norm,_,_ = min_max_normalize(X_val)
    # X_test_norm,_,_ = min_max_normalize(X)

    # checkpoint = keras.callbacks.ModelCheckpoint('bestW.weights.h5', monitor='val_loss', verbose=False,save_weights_only=True, save_best_only=True)
    # early_stop = EarlyStopping(monitor="val_loss", patience=500, verbose=True, restore_best_weights=True, mode='min')
    # callbacks_list = [early_stop,checkpoint]
    # Modell erstellen und trainieren
    print(X.shape[1], neurons_per_layer,layers,l2rate,learn, activation_function, X_val_norm.shape, Y_val.shape)
    model = create_model(neurons_per_layer, layers, activation_function, input_dim=X.shape[1],l2rate=l2rate,learning=learn,opimizer=opt)
    model.save('StartANN.h5')

    checkpoint = keras.callbacks.ModelCheckpoint('bestW.weights.h5', monitor='val_loss', verbose=False,save_weights_only=True, save_best_only=True)
    early_stop = EarlyStopping(monitor="val_loss", patience=500, verbose=True, restore_best_weights=True, mode='min')
    callbacks_list = [early_stop,checkpoint]

    # Training
    # model.summary()

    history = model.fit(
        X_train_norm,
        Y_train,
        validation_data=(X_val_norm, Y_val),
        callbacks=callbacks_list,
        epochs=epochs,
        verbose=0,
    )
    y_pred = (model.predict(X_test_norm) > 0.5).astype(int)

    # Genauigkeit berechnen
    accuracy = np.mean(Y_test == y_pred.flatten())
    print("Accuracy:", accuracy)


    myANN = load_model('StartANN.h5')
    myANN.load_weights('bestW.weights.h5')
    # Vorhersagen und Evaluation
    y_pred = (myANN.predict(X_test_norm) > 0.5).astype(int)

    # Genauigkeit berechnen
    accuracy = np.mean(Y_test == y_pred.flatten())
    print("Accuracy:", accuracy)

    if save_model:
        model.save("trained_model.h5")
        return f"Model saved as 'trained_model.h5'", X_test_norm, Y_test

    return "Training complete. Model not saved.", X_test_norm, Y_test


# Modell laden und Vorhersagen treffen
def load_and_predict(model_path, input_data, xtest,ytest):
    # if not os.path.exists(model_path):
    #     return "Model file not found."

    # model = tf.keras.models.load_model(model_path)
    model = load_model('StartANN.h5')
    model.load_weights('bestW.weights.h5')
    if not input_data:
        prediction = (model.predict(xtest) > 0.5).astype(int)
        accuracyV = np.mean(ytest == prediction.flatten())
        print("Accuracy:", accuracyV)
        return  f"Accuracy{accuracyV} \n Prediction: {prediction,ytest}"
        # return f"Accuracy: {accuracyV}\nPredictions:\n{list(zip(ytest, prediction.flatten()))}"

    try:
        # Eingabedaten (z. B. '1,2,3') in eine Liste von Indizes umwandeln
        indices = [int(i.strip()) for i in input_data.split(",")]
    except ValueError:
        return "Invalid input data. Please provide a comma-separated list of integers."

    # Sicherstellen, dass Indizes gültig sind
    if any(idx < 0 or idx >= len(xtest) for idx in indices):
        return f"Invalid index. Please provide indices between 0 and {len(xtest) - 1}."

    # Beispiele aus xtest und ytest extrahieren
    xe = xtest[indices]
    ye = ytest[indices]

    # Vorhersagen für die ausgewählten Beispiele treffen
    prediction = (model.predict(xe) > 0.5).astype(int)

    # Genauigkeit für die ausgewählten Daten berechnen
    accuracyV = np.mean(ye == prediction.flatten())

    # Ergebnis formatieren
    result = f"Accuracy for selected examples: {accuracyV}\n"
    for idx, x, y, p in zip(indices, xe, ye, prediction.flatten()):
        result += f"Example {idx}:\n  Input: {x}\n  True: {y}\n  Predicted: {p}\n"

    return result

        # prediction = (model.predict(xtest)> 0.5).astype(int)
        # input_array = np.array(input_data).reshape(1, -1)
        # xe = xtest[input_array]
        # ye = ytest[input_array]
        # prediction = (model.predict(xe)> 0.5).astype(int)
        # ytest = ye

    # return f"MSE: {mean_squared_error(y_true=ytest,y_pred=prediction)} \n Prediction: {prediction,ytest}"
    # accuracyV = np.mean(ytest == prediction.flatten())
    # print("Accuracy:", accuracyV)
    # return  f"Accuracy{accuracyV} \n Prediction: {prediction,ytest}"  #Validierungsmengengenauigkeit noch ausgeben


# GUI mit Gradio erstellen
def filetransform(file):
    print("TEST2")
    """Lädt eine CSV-Datei und gibt die Daten als numpy-Array zurück."""
    data = np.loadtxt(file, delimiter=",", skiprows=1)  # CSV-Datei ohne Header einlesen
    return data


def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("### Neuronales Netzwerk Training & Vorhersage")

        xtest_state = gr.State()
        ytest_state = gr.State()

        with gr.Tab("Training"):
            # neurons_input = gr.Textbox(label="Neuronen pro Schicht (z.B. 32, 64, 16)", value="")
            neurons_input = gr.Number(label="Neuronen in der ersten Layer- Kegel", value=64)
            layers = gr.Number(label="Layers", minimum=2, value=2)
            activation_input = gr.Dropdown(["relu", "tanh", "sigmoid","elu","softmax"], label="Aktivierungsfunktion", value="sigmoid")
            l2reg = gr.Number(label='L2-Regularisierungswert',value=0.0001,maximum=1,step=0.00001)
            learning = gr.Number(label='Learning Rate',value=0.1,maximum=1,step=0.001)
            optimizer = gr.Dropdown(['SGD','Adam','RMSprop'],label='Optimizer',value='Adam')
            train_split_input = gr.Slider(0.0, 1, step=0.01, label="Val/Test Größe", value=0.15)
            epochs_input = gr.Slider(0, 2000, step=10, label="Epochs", value=100)
            save_model_checkbox = gr.Checkbox(label="Modell speichern?", value=True)
            train_button = gr.Button("Training starten")
            train_output = gr.Textbox(label="Status")
            file_input = gr.File(label="Dataset hochladen (CSV mit Header)", file_types=[".csv"])

            train_button.click(
                train_model,
                inputs=[neurons_input, layers, activation_input, train_split_input, epochs_input, save_model_checkbox,
                        file_input,l2reg,learning,optimizer],
                outputs=[train_output, xtest_state, ytest_state],
            )

        with gr.Tab("Vorhersage"):
            model_path_input = gr.Textbox(label="Pfad zum Modell", value="trained_model.h5")
            input_data_input = gr.Textbox(label="Eingabedaten (z.B. 0.1, 0.2, ...)",
                                          value="")
            predict_button = gr.Button("Vorhersage starten")
            prediction_output = gr.Textbox(label="Vorhersage")

        predict_button.click(
            load_and_predict,
            inputs=[model_path_input, input_data_input, xtest_state,ytest_state],
            outputs=prediction_output,
        )

        return demo


# Starten der Gradio-App
app = gradio_interface()
app.launch()
