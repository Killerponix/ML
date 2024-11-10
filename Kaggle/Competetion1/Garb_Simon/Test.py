import math

from sklearn.model_selection import KFold
import numpy as np


class knnRegression:
    def fit(self,X,Y,mode=1):
        if mode ==1:
            self.xMin= X.min(axis=0)
            self.xMax=X.max(axis=0)
            self.XTrain = (X-self.xMin)/(self.xMax-self.xMin)
            self.YTrain=Y
        elif mode==2:
            self.xMean = X.mean(axis=0)
            self.xstd = X.std(axis=0)
            self.XTrain = (X-self.xMean)/self.xstd
            self.YTrain = Y
        else:
            self.xMedian = np.median(X, axis=0)
            self.xIQR = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
            self.XTrain = (X - self.xMedian) / self.xIQR
            self.YTrain = Y
    # def predict(self,X,k=3,smear=1,norm=None,kernel=3):
    #     if hasattr(self,"xMin"):
    #         X=(X-self.xMin)/(self.xMax-self.xMin)
    #     elif hasattr(self,"xMean"):
    #         X= (X-self.xMean)/self.xstd
    #     predictions = []
    #     for p in X:
    #         diff = self.XTrain - p
    #         dist = np.linalg.norm(diff, axis=1, ord=norm)
    #         # Finde die k nächsten Nachbarn
    #         knearest = np.argpartition(dist, k)[:k]
    #         distsum = np.sum(1/(dist[knearest]+smear/k))
    #         if kernel== 0:
    #             weights=1/(dist+smear/k)
    #         elif kernel==1:
    #             weights= np.exp(-dist**2/(2*(smear)))
    #         elif kernel==2:
    #             weights=np.ones_like(dist)
    #         # elif kernel==4:
    #         #     weights = np.exp(-dist[knearest]**2 / (2 * (smear/k)**2))
    #         else:
    #             weights = (1/(dist[knearest]+smear/k))/distsum
    #
    #         # weights_sum = np.sum(weights,axis=1,keepdims=True)
    #         # weights/=weights_sum
    #         # Durchschnitt der Zielwerte der nächsten Nachbarn als Vorhersagewert
    #         y_pred = np.sum(weights*self.YTrain[knearest],axis=1)#mean?
    #         predictions.append(y_pred)
    #     return np.array(predictions)

    def predict(self, X, k=3, smear=1, norm=None, kernel=3):
        if hasattr(self, "xMin"):
            X = (X - self.xMin) / (self.xMax - self.xMin)
        elif hasattr(self, "xMean"):
            X = (X - self.xMean) / self.xstd
        elif hasattr(self, "xMedian"):
            X = (X - self.xMedian) / self.xIQR

        predictions = []
        for p in X:
            diff = self.XTrain - p
            dist = np.linalg.norm(diff, axis=1, ord=norm)
            knearest = np.argpartition(dist, k)[:k]

            # Berechnung von distsum ohne axis-Spezifikation, um 1D-Ergebnis zu behalten
            # distsum = np.sum(1 / (dist[knearest] + smear / k))
            distsum= np.sum(1/dist+smear/k)
            distsum= np.repeat(distsum,k)
            weights = None


            if kernel == 0:
                weights = 1 / (dist[knearest] + smear / k)
            elif kernel == 1:
                weights = np.exp(-dist[knearest]**2 / (2 * smear))
            elif kernel == 2:
                weights = np.ones_like(dist[knearest])
            elif kernel == 4:
                weights = np.exp(-dist[knearest]**2 / (2 * (smear / k)**2))
            elif kernel == 5:
                # Berechne y_pred direkt bei kernel=5 und füge der Prediction hinzu
                y_pred = np.sum((1 / (dist[knearest] + smear / k)) * self.YTrain[knearest])
                predictions.append(y_pred)
                return np.array(predictions)
            elif kernel==6:
                weights = np.mean((-dist[knearest]+smear/k)*self.YTrain[knearest]/(2*smear/k+np.sum(dist[knearest]*smear)))
            elif kernel==7:
                weights = np.mean((-dist[knearest]+smear/k)*self.YTrain[knearest]/(2*smear/k))/distsum
            elif kernel==8:
                weights=np.mean((-dist[knearest]+smear/k*self.YTrain[knearest])/distsum**2)
            elif kernel==9:
                weights=np.exp(np.dot(dist[knearest]-smear/k,distsum)*self.YTrain[knearest])
            elif kernel == 10:
                weights = np.exp(-dist[knearest] ** 2 / (2 * (smear * k) ** 2)) / distsum
            elif kernel==11:
                weights=np.exp(dist[knearest]*smear/distsum*k)*1/k
            else:
                weights = np.sum(smear+dist[knearest]/k)
            #Gewichte normalisieren
            # weights_sum = np.sum(weights)
            # weights /= weights_sum

            # Berechne den gewichteten Mittelwert der k-Nachbarn
            y_pred = np.sum(weights * self.YTrain[knearest])
            predictions.append(y_pred)

        return np.array(predictions)





def train_test_split(X, y, test_size=0.2, shuffle=True):
    """
    Splits the dataset into training and testing sets using numpy arrays.

    :param X: Numpy array of features.
    :param y: Numpy array of labels.
    :param test_size: Float representing the proportion of the dataset to include in the test split.
    :param shuffle: Boolean indicating whether to shuffle the data before splitting (default True).
    :return: Four numpy arrays: X_train, X_test, y_train, y_test
    """
    # Combine X and y to shuffle them together
    data = np.column_stack((X, y))
    np.random.seed(27)
    if shuffle:
        np.random.shuffle(data)  # Shuffle the combined data

    # Calculate the index at which to split the data
    split_index = int(len(data) * (1 - test_size))

    # Split the data into training and testing sets
    train_data = data[:split_index]
    test_data = data[split_index:]

    # Separate the features (X) and labels (y)
    X_train = train_data[:, :-1]  # All columns except the last
    y_train = train_data[:, -1]  # Last column
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    return X_train, X_test, y_train, y_test

def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)  #2
    # print(f"MSE: {mse:.4f}")
    return mse


def cross_val_knn(X, y, model, k_values,mode, smear_values, kernel_values, norm_values, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=27)
    best_mse = float('inf')
    best_params = {}
    for mods in mode:
        for ke in kernel_values:
            for norms in norm_values:
                for sm in smear_values:
                    for k in k_values:
                        mse_scores = []

                        for train_index, val_index in kf.split(X):
                            X_train, X_val = X[train_index], X[val_index]
                            y_train, y_val = y[train_index], y[val_index]

                            # Trainiere das Modell
                            model.fit(X_train, y_train,mode=mode )

                            # Mache Vorhersagen
                            y_pred = model.predict(X_val, k=k, smear=sm/10, kernel=ke, norm=norms)

                            # Berechne den MSE und füge ihn zur Liste hinzu
                            mse_scores.append(mean_squared_error(y_val, y_pred))

                        # Durchschnittlichen MSE berechnen
                        avg_mse = np.mean(mse_scores)

                        # print(f"K: {k}, Smear: {sm/10},mode: {mods}, Kernel: {ke}, Norm: {norms}, Avg MSE:"
                        #       f" {avg_mse:.4f}")

                        # Wenn dies die beste MSE ist, dann speichere die Parameter
                        if avg_mse < best_mse:
                            best_mse = avg_mse
                            best_params = {
                                'k': k,
                                'smear': sm/10,
                                'kernel': ke,
                                'norm': norms,
                                'mode': mods
                            }
                            print(f"Neue Best: K: {k}, Smear: {sm/10},mode: {mods}, Kernel: {ke}, Norm: {norms}, "
                                  f"Avg MSE:"
                                  f" {avg_mse:.4f}")

    print(f"Beste Parameter: {best_params}, Beste MSE: {best_mse:.4f}")
    return best_params, best_mse

# Beispieldurchführung
if __name__ == '__main__':
    # Daten laden
    XTrain = np.load("XTrain.npy")
    YTrain = np.load("YTrain.npy")

    # Modell und Parameterbereiche definieren
    knn = knnRegression()
    mode= range(1,2)
    k_values = range(1, 21)          # k-Werte von 1 bis 20
    smear_values = range(0,10)       # smear-Werte von 1 bis 2
    kernel_values =[11]      # kernel-Werte von 3 bis 4
    norm_values = [None, 1]       # norm-Werte: None, 1, 2

    # Kreuzvalidierung durchführen
    # best_params, best_mse = cross_val_knn(XTrain, YTrain, knn, k_values,mode, smear_values, kernel_values, norm_values)

    # Nach der besten Parameterkombination das Modell trainieren
    # knn.fit(XTrain, YTrain, best_params['mode'])
    XTest = np.load("XTest.npy")
    # y_predict = knn.predict(XTest, k=best_params['k'], smear=best_params['smear']/10,
    #                         kernel=best_params['kernel'], norm=best_params['norm'])
    knn.fit(XTrain,YTrain,1)
    y_predict = knn.predict(XTest,3,0,1,11)
    # knn.fit(XTrain, YTrain, best_params['mode'])
    XTest = np.load("XTest.npy")
    # y_predict = knn.predict(XTest, k=best_params['k'], smear=best_params['smear'],
    #                         kernel=best_params['kernel'], norm=best_params['norm'])

    # Ergebnisse exportieren
    import csv
    filename = 'yPredict_Garb_Simon_cross.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "predicted"])
        for idx, prediction in enumerate(y_predict, start=1):
            writer.writerow([idx, prediction])

    print(f"CSV-Datei '{filename}' erfolgreich erstellt!")
