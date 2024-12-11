import gradio as gr
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from Programme.Model.util.mse import mean_squared_error
import os


# Funktion zum Erstellen des Modells
def create_model(neurons_per_layer, layers, activation_function, input_dim):
    print(input_dim, neurons_per_layer, activation_function)
    # Eingabedimension überprüfen
    if not isinstance(input_dim, int):
        raise ValueError(f"Invalid input_dim: {input_dim}. Must be an integer.")

    model = Sequential()

    # Definieren der Eingabeform explizit
    # model.add(Dense(neurons_per_layer[0], activation=activation_function, input_dim=input_dim))
    model.add(Dense(neurons_per_layer, activation="sigmoid", input_dim=8))
    neurons = neurons_per_layer
    # Hinzufügen der versteckten Schichten
    for i in range(layers):
        model.add(Dense(neurons, activation=activation_function))
        neurons = neurons // 2

    # Ausgabe-Schicht
    model.add(Dense(1, activation='linear'))

    # Zusammenfassung und Kompilierung
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


def divTestTrainSet(X, Y, size):
    ValSet = np.random.choice(X.shape[0], int(X.shape[0] * size), replace=False)
    TrainSet = np.delete(np.arange(0, Y.shape[0]), ValSet)
    XVal = X[ValSet, :]
    YVal = Y[ValSet]
    X = X[TrainSet, :]
    Y = Y[TrainSet]
    return (XVal, YVal, X, Y)


# Training des Modells
def train_model(neurons_per_layer, layers, activation_function, train_split, epochs, save_model, data_file):
    if data_file is not None:
        # Datei verarbeiten und Daten laden
        print("TEST")
        data = filetransform(data_file)
        # X, Y = data[:,:-1], data[:,-1:]
        X, Y = data[:, :-1], data[:, -1]
        xtest, ytest, xtrain, ytrain = divTestTrainSet(X, Y, 0.2)

        print(Y)
    else:
        # Dummy-Daten erstellen, falls keine Datei hochgeladen wurde
        X = np.random.random((1000, 10))
        Y = np.random.randint(2, size=(1000, 1))
        xtest, ytest, xtrain, ytrain = divTestTrainSet(X, Y, 0.2)

    # Daten in Training und Validierung aufteilen
    split_index = int(train_split * len(xtrain))
    train_data, val_data = xtrain[:split_index], xtrain[split_index:]
    train_labels, val_labels = ytrain[:split_index], ytrain[split_index:]

    # Modell erstellen und trainieren
    print(X.shape[1], neurons_per_layer, activation_function, val_data.shape, val_labels.shape)
    model = create_model(neurons_per_layer, layers, activation_function, input_dim=X.shape[1])
    history = model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        epochs=epochs,
        verbose=1,
    )
    if save_model:
        model.save("trained_model.h5")
        return f"Model saved as 'trained_model.h5'", xtest, ytest

    return "Training complete. Model not saved.", xtest, ytest


# Modell laden und Vorhersagen treffen
def load_and_predict(model_path, input_data, xtest,ytest):
    if not os.path.exists(model_path):
        return "Model file not found."

    model = tf.keras.models.load_model(model_path)
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(xtest)
    return f"MSE: {mean_squared_error(y_true=ytest,y_pred=prediction)} \n Prediction: {prediction,ytest}"


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
            layers = gr.Number(label="Layers", minimum=2, value=3)
            activation_input = gr.Dropdown(["relu", "tanh", "sigmoid","elu","softmax"], label="Aktivierungsfunktion", value="relu")
            train_split_input = gr.Slider(0.0, 1, step=0.1, label="Train/Test Aufteilung", value=0.8)
            epochs_input = gr.Slider(0, 2000, step=10, label="Epochs", value=100)
            save_model_checkbox = gr.Checkbox(label="Modell speichern?", value=True)
            train_button = gr.Button("Training starten")
            train_output = gr.Textbox(label="Status")
            file_input = gr.File(label="Dataset hochladen (CSV mit Header)", file_types=[".csv"])

            train_button.click(
                train_model,
                inputs=[neurons_input, layers, activation_input, train_split_input, epochs_input, save_model_checkbox,
                        file_input],
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
