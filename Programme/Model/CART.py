import numpy as np
from binaryTree import tree

class bRegressionTree:
    def _calLRSS(self,y):
        yMean=np.sum(y)/len(y)
        L2=np.sum((y-yMean)**2)
        return(L2)

    # def _bestSplit(self,X,y,feature):
    #     RSS=np.inf
    #     bestSplit=np.inf
    #     XSort=np.unique(X[:,feature].round(self.xDecimals))
    #     XDiff=(XSort[1:len(XSort)]+XSort[0:len(XSort)-1])/2
    #     for i in range(XDiff.shape[0]):
    #         index=np.less(X[:,feature],XDiff[i])
    #     if not (np.all(index) or np.all(~index)):
    #         RSS_1=self._calLRSS(y[index])
    #         RSS_2=self._calLRSS(y[~index])
    #         RSSSplit=RSS_1+RSS_2
    #         if RSS>RSSSplit:
    #             RSS=RSSSplit
    #             bestSplit=XDiff[i]
    #     return (bestSplit,RSS)

    def _bestSplit(self, X, y, feature):
        RSS = np.inf
        bestSplit = np.inf
        XSort = np.unique(X[:, feature].round(self.xDecimals))
        XDiff = (XSort[1:] + XSort[:-1]) / 2  # Mittlere Werte zwischen den aufeinanderfolgenden Sortierten

        for i in range(XDiff.shape[0]):  # Hier Schleife über die potenziellen Trennpunkte
            index = np.less(X[:, feature], XDiff[i])

            if not (np.all(index) or np.all(~index)):  # Sicherstellen, dass Split valide ist
                RSS_1 = self._calLRSS(y[index])
                RSS_2 = self._calLRSS(y[~index])
                RSSSplit = RSS_1 + RSS_2

                if RSS > RSSSplit:
                    RSS = RSSSplit
                    bestSplit = XDiff[i]

        return bestSplit, RSS



    def _chooseFeature(self,X,y):
        G         = np.zeros(X.shape[1])
        bestSplit = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            ( bestSplit[i] , G[i] ) = self._bestSplit(X,y,i)
        smallest = np.argmin(G)
        return G[smallest], bestSplit[smallest], smallest

    def _ComputeValue(self,y):
        return (np.sum(y)/len(y))

    def __init__(self, threshold=0.1, xDecimals=8, minLeafNodeSize=3, max_depth=None):
        self.bTree = None
        self.threshold = threshold
        self.xDecimals = xDecimals
        self.minLeafNodeSize = minLeafNodeSize
        self.max_depth = max_depth

    def _GenTree(self, X, y, parentNode, branch, depth=0):
        commonValue = self._ComputeValue(y)
        initG = self._calLRSS(y)
        if  initG < self.threshold or X.shape[0] <= self.minLeafNodeSize or (self.max_depth is not None and depth >= self.max_depth):
            self.bTree.addNode(parentNode, branch, commonValue)
            return

        (G, bestSplit, chooseA) = self._chooseFeature(X, y)
        if initG < self.threshold or X.shape[0] <= self.minLeafNodeSize:
            self.bTree.addNode(parentNode, branch, commonValue)
            return

        if parentNode == None:
            self.bTree = tree(chooseA, bestSplit, '<')
            myNo = 0
        else:
            myNo = self.bTree.addNode(parentNode, branch, bestSplit, operator='<', varNo=chooseA)

        index = np.less(X[:, chooseA], bestSplit)
        XTrue = X[index, :]
        yTrue = y[index]
        XFalse = X[~index, :]
        yFalse = y[~index]

        if XTrue.shape[0] > self.minLeafNodeSize:
            self._GenTree(XTrue, yTrue, myNo, True, depth+1)
        else:
            commonValue = self._ComputeValue(yTrue)
            self.bTree.addNode(myNo, True, commonValue)
        if XFalse.shape[0] > self.minLeafNodeSize:
            self._GenTree(XFalse, yFalse, myNo, False, depth+1)
        else:
            commonValue = self._ComputeValue(yFalse)
            self.bTree.addNode(myNo, False, commonValue)
        return()


    def fit(self, X,y):
        self._GenTree(X,y,None,None)

    def predict(self, X):
        return(self.bTree.eval(X))

    def decision_path(self, X):
        return(self.bTree.trace(X))

    def weightedPathLength(self,X):
        return(self.bTree.weightedPathLength(X))

    def numberOfLeafs(self):
        return(self.bTree.numberOfLeafs())

import random
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
    y_train = train_data[:, -1]   # Last column
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    return X_train, X_test, y_train, y_test
def calculate_accuracy(y_true, y_pred):
    """
    Calculates the accuracy of predictions.

    :param y_true: List or array of actual (true) labels.
    :param y_pred: List or array of predicted labels.
    :return: Accuracy as a float.
    """
    # Count the number of correct predictions
    correct = sum(true == pred for true, pred in zip(y_true, y_pred))

    # Calculate the accuracy
    accuracy = correct / len(y_true)

    # Print the accuracy as a percentage
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return accuracy

def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 1) #2
    print(f"MSE: {mse:.4f}")
    return mse



if __name__ == '__main__':
    import sys
    sys.setrecursionlimit(10000)
    XTrain = np.load("XTrain.npy")
    XTest = np.load("XTest.npy")
    YTrain=np.load("YTrain.npy")
    X_train , X_test, y_train, y_test = train_test_split(XTrain,YTrain)
    bestminleaf=20
    bestdepth=4
    bestmse=1000000
    bestthres=0.5
    for i in range(10,21):
        for j in range(1,6):
            for l in range(1,20):
                myTree = bRegressionTree(max_depth=i,minLeafNodeSize=j,threshold=l/10)
                myTree.fit(X_train,y_train)
                yPredict = myTree.predict(X_test)
                tmp = mean_squared_error(y_test,yPredict)
                print("Aktuelle: ",i,j,l/10)
                if bestmse > tmp:
                    bestmse=tmp
                    bestminleaf=j
                    bestdepth=i
                    bestthres=l/10

    print(bestmse,bestdepth,bestminleaf)
    bestDec=5

    # for i in range(1,10):
    #     for j in range(1,11):
    #         testTree = bRegressionTree(max_depth=bestdepth,minLeafNodeSize=bestminleaf,xDecimals=i,threshold=j/100)
    #         testTree.fit(X_train,y_train)
    #         ypred = testTree.predict(X_test)
    #         tmp = mean_squared_error(y_test,ypred)
    #         print("Aktuelle: ",i,j/100)
    #         if bestmse > tmp:
    #             bestmse=tmp
    #             bestDec=i
    #             bestthres=j/100
    # print(bestDec,bestthres,bestmse)





    myTree = bRegressionTree(max_depth=bestdepth,minLeafNodeSize=bestminleaf,xDecimals=bestDec,threshold=bestthres) #20 und 3?
    myTree.fit(XTrain,YTrain)
    y_predict = [...]
    y_predict = myTree.predict(XTest)
    # myTree.fit(X_train,y_train)
    # yPredict = myTree.predict(X_test)
    #calculate_accuracy(y_test,yPredict)
    # mean_squared_error(y_test,yPredict)
    # print(yPredict)

# print(sys.getrecursionlimit())


import csv
# Beispiel-Daten für Vorhersagen, ersetze dies mit deinem eigenen 'y_predict'
  # z.B., [0, 1, 0, ..., 1] (1634 Werte)

# Überprüfen, ob die Länge der Vorhersagen 1634 ist
#assert len(y_predict) == 1634, f"Es müssen 1634 Vorhersagen sein, aber es gibt {len(y_predict)}."

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





