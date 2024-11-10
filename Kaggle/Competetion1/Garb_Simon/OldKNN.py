from scipy.spatial import KDTree


class knnRegression:
    def fit(self, X, Y, scaler=0):
        if scaler == 0:
            self.xMin = X.min(axis=0)
            self.xMax = X.max(axis=0)

        self.xMin = X.min(axis=0)
        self.xMax = X.max(axis=0)
        self.XTrain = (X - self.xMin) / (self.xMax - self.xMin)
        self.kdTree = KDTree(self.XTrain)
        self.YTrain = Y

    def predict(self, X, k=3, smear=1):
        X = (X - self.xMin) / (self.xMax - self.xMin)
        (dist, neighbours) = self.kdTree.query(X, k)
        distsum = np.sum(1 / (dist + smear / k), axis=1)
        distsum = np.repeat(distsum[:, None], k, axis=1)
        dist = (1 / distsum) * 1 / (dist + smear / k)
        y = np.sum(dist * self.YTrain[neighbours], axis=1)
        return (y)