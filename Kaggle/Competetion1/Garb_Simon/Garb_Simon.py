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

            distsum = np.sum(1 / (dist[knearest] + smear / k))
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
            else:
                weights = (1 / (dist[knearest] + smear / k)) / distsum

            #Gewichte normalisieren
            weights_sum = np.sum(weights)
            weights /= weights_sum

            # Berechne den gewichteten Mittelwert der k-Nachbarn
            y_pred = np.sum(weights * self.YTrain[knearest])
            predictions.append(y_pred)

        return np.array(predictions)

if __name__ == '__main__':
    # Daten laden
    knn = knnRegression()
    XTrain = np.load("XTrain.npy")
    YTrain = np.load("YTrain.npy")
    XTest = np.load("XTest.npy")
    knn.fit(XTrain,YTrain,1)
    y_predict = knn.predict(XTest,10,1,1,4)
    import csv
    filename = 'y_Predict_Garb_Simon.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "predicted"])
        for idx, prediction in enumerate(y_predict, start=1):
            writer.writerow([idx, prediction])

    print(f"CSV-Datei '{filename}' erfolgreich erstellt!")
