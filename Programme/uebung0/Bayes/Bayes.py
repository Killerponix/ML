import numpy as np


class BayesK:

    def __init__(self, continuous=None):
        self.continuous = continuous

    def fit(self, X, Y):
        self.classes, self.P = np.unique(Y, return_counts=True)
        self.P = self.P / X.shape[0]

        if self.continuous is None:
            self.continuous = np.zeros(X.shape[1], dtype=bool)
        Xd = X[:, ~self.continuous]
        Xc = X[:, self.continuous]

        self.noOffFeatureC = Xc.shape[1]
        self.noOffFeatureD = Xd.shape[1]

        FCMax = 0
        self.featurecategories = []
        for i in range(self.noOffFeatureC):
            self.featurecategories.append(np.unique(Xd[:, i]))
            FCMax = max(FCMax, len(self.featurecategories[i]))

        self.PP = np.zeros((len(self.classes), self.noOffFeatureD, FCMax))
        for k in range(self.noOffFeatureD):
            for i, c in enumerate(self.classes):
                for j, f in enumerate(self.featurecategories[k]):
                    xk = (Xd[:, k] == f)
                    theClass = (Y == c)
                    self.PP[i, k, j] = np.sum(xk & theClass) / np.sum(theClass)

        self.mu = np.zeros((len(self.classes), self.noOffFeatureC))
        self.sigma = np.zeros((len(self.classes), self.noOffFeatureC))
        for k in range(self.noOffFeatureC):
            for i, c in enumerate(self.classes):
                self.mu[i, k] = np.mean(Xc[Y == c, k])
                self.sigma[i, k] = np.std(Xc[Y == c, k])

    def GaussDistibution(self, x, mu, sigma):
        y = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        return y

    def predictProba(self, X):
        if len(X.shape) == 1: X = X[np.newaxis, :]
        Xd = X[:, ~self.continuous]
        Xc = X[:, self.continuous]

        # Product k=1:m P(x^(k)|i)*P(i)
        Product = np.ones((X.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes):
            for k in range(self.noOffFeatureD):
                indK = np.searchsorted(self.featurecategories[k], Xd[:, k])
                Product[:, i] *= self.PP[i, k, indK]
            for k in range(self.noOffFeatureC):
                Product[:, i] *= self.GaussDistibution(Xc[:, k], self.mu[i, k], self.sigma[i, k])
        Denominator = Product @ self.P
        PofClass = self.P * Product / Denominator[:, np.newaxis]
        return PofClass

    def predict(self, X):
        chosenClass = np.argmax(self.predictProba(X), axis=1)
        return self.classes[chosenClass]
