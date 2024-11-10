import numpy as np


class knnRegression:
    def fit(self,X,Y,mode=1):
        if mode ==1:
            self.xMin= X.min(axis=0)
            self.xMax=X.max(axis=0)
            self.XTrain = (X-self.xMin)/(self.xMax-self.xMin)
        else:
            self.xMean = X.mean(axis=0)
            self.xstd = X.std(axis=0)
            self.XTrain = (X-self.xMean)/self.xstd
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
        if hasattr(self,"xMin"):
            X=(X-self.xMin)/(self.xMax-self.xMin)
        elif hasattr(self,"xMean"):
            X= (X-self.xMean)/self.xstd

        predictions = []
        for p in X:
            diff = self.XTrain - p
            dist = np.linalg.norm(diff, axis=1, ord=norm)
            knearest = np.argpartition(dist, k)[:k]

            distsum = np.sum(1 / (dist[knearest] + smear / k))

            if kernel == 0:
                weights = 1 / (dist[knearest] + smear / k)
            elif kernel == 1:
                weights = np.exp(-dist[knearest]**2 / (2 * smear))
            elif kernel == 2:
                weights = np.ones_like(dist[knearest])
            elif kernel == 4:
                weights = np.exp(-dist[knearest]**2 / (2 * (smear / k)**2))
            else:
                weights = (1 / (dist[knearest] + smear / k)) / distsum

            weights_sum = np.sum(weights)
            weights /= weights_sum  # Normalisiere die Gewichte

            # Berechne den gewichteten Mittelwert
            y_pred = np.sum(weights * self.YTrain[knearest])  # Vorhersagewert
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
    np.random.seed(42)
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


from sklearn.model_selection import KFold
def cross_val_knn(X, y, model, k_values, smear_values, kernel_values, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_mse = float('inf')
    for k in k_values:
        for smear in smear_values:
            for kernel in kernel_values:
                mse_scores = []
                for train_index, val_index in kf.split(X):
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = y[train_index], y[val_index]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val, k=k, smear=smear, kernel=kernel)
                    mse_scores.append(mean_squared_error(y_val, y_pred))
                avg_mse = np.mean(mse_scores)
                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_params = {'k': k, 'smear': smear, 'kernel': kernel}
    return best_params, best_mse
def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)  #2
    print(f"MSE: {mse:.4f}")
    return mse

if __name__ == '__main__':
    import sys

    sys.setrecursionlimit(10000)
    XTrain = np.load("XTrain.npy")
    XTest = np.load("XTest.npy")
    YTrain = np.load("YTrain.npy")
    knn = knnRegression()
    bestmse = 1000
    bestk = 3
    best_i = 1
    bestsm=1
    bestke=4
    bestnorm=None
    X_train, X_test, y_train, y_test = train_test_split(XTrain, YTrain)
    for ke in range(3,5):
        for norms in range(None,2):
            for sm in range(1,2):
                for k in range(0, 20):
                    for i in range(0, 25):
                        knn.fit(X_train, y_train,sm)
                        y_predict = knn.predict(X_test, k, i,norm=norms,kernel=ke)
                        acc = mean_squared_error(y_test, y_predict)
                        print("Aktuelle K und I: ", k, i, "Im Modus: ",sm,"im Kernel: ",ke," norm: ",norms)
                        if bestmse > acc:
                            bestmse = acc
                            bestk = k
                            best_i = i
                            bestsm=sm
                            bestnorm=norms
    print(bestmse, " wurde mit ", bestk, best_i, " erreicht im Modus: ",bestsm, "und Kernel:",bestke,"norm: ",bestnorm)

knn.fit(XTrain, YTrain,bestsm)
y_predict = [...]
y_predict = knn.predict(XTest,k=bestk, smear=best_i,norm=bestnorm,kernel=bestke)
# y_predict = plainKNNRegressor(XTrain,YTrain,XTest,2,None)
assert len(y_predict) == 1634, f"Es müssen 1634 Vorhersagen sein, aber es gibt {len(y_predict)}."

import csv

filename = 'yPredict_Garb_Simon.csv'
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "predicted"])
    for idx, prediction in enumerate(y_predict, start=1):
        writer.writerow([idx, prediction])

print(f"CSV-Datei '{filename}' erfolgreich erstellt!")
