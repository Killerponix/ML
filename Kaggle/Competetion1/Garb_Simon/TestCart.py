import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from CART import bRegressionTree
from CART import train_test_split


# Dein bereits vorhandenes bRegressionTree-Modell
class bRegressionTreeWrapper:
    def __init__(self, max_depth=None, minLeafNodeSize=1, threshold=0.1, xDecimals=8):
        self.model = bRegressionTree(max_depth=max_depth, minLeafNodeSize=minLeafNodeSize, threshold=threshold,
                                     xDecimals=xDecimals)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


# Füge eine Funktion für die Grid Search hinzu
def grid_search_optimization(X_train, y_train):
    # Erstelle eine Instanz der Wrapper-Klasse
    model = bRegressionTreeWrapper()

    # Definiere den Parameterbereich, den du durchsuchen möchtest
    param_grid = {
        'max_depth': [10, 15, 20, 25],  # Teste verschiedene Baumtiefen
        'minLeafNodeSize': [3, 5, 7, 10],  # Teste verschiedene minimale Blattgrößen
        'threshold': [0.01, 0.05, 0.1, 0.5],  # Teste verschiedene RSS-Schwellenwerte
        'xDecimals': [6, 8, 10]  # Teste verschiedene Genauigkeiten der Trennpunkte
    }

    # Definiere den Scorer basierend auf dem negativen MSE
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # Erstelle das GridSearchCV-Objekt
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=mse_scorer, cv=5, verbose=1)

    # Führe die Grid Search aus
    grid_search.fit(X_train, y_train)

    # Gebe die besten Parameter und die beste Leistung aus
    print("Beste Parameterkombination:", grid_search.best_params_)
    print("Beste MSE:", -grid_search.best_score_)

    return grid_search.best_estimator_


# Beispiel Hauptprogramm
if __name__ == '__main__':
    import sys

    sys.setrecursionlimit(10000)

    # Lade deine Daten
    XTrain = np.load("XTrain.npy")
    XTest = np.load("XTest.npy")
    YTrain = np.load("YTrain.npy")

    # Splitte die Daten in Training und Test
    X_train, X_test, y_train, y_test = train_test_split(XTrain, YTrain)

    # Führe Grid Search aus, um die besten Hyperparameter zu finden
    best_model = grid_search_optimization(X_train, y_train)

    # Vorhersagen mit dem besten Modell
    y_predict = best_model.predict(XTest)

    # Berechne den Mean Squared Error auf den Testdaten
    mse = mean_squared_error(y_test, y_predict)
    print(f"Mean Squared Error auf Testdaten: {mse:.4f}")

    # Überprüfe die Länge der Vorhersagen
    assert len(y_predict) == 1634, f"Es müssen 1634 Vorhersagen sein, aber es gibt {len(y_predict)}."

    # Speichere die Vorhersagen in eine CSV-Datei
    import csv

    # Der Dateiname
    filename = 'yPredict_Garb_Simon.csv'

    # Schreibe die CSV-Datei
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Schreibe die Kopfzeile
        writer.writerow(["id", "predicted"])

        # Schreibe die Vorhersagedaten
        for idx, prediction in enumerate(y_predict, start=1):
            writer.writerow([idx, prediction])

    print(f"CSV-Datei '{filename}' erfolgreich erstellt!")
