import numpy as np
from CARTDecisionTreeRF import bDecisionTree

class randomForestDecision:
    def __init__(self,max_Features=0,noOfTrees=10,threshold = 10**-8, xDecimals = 8, minLeafNodeSize=3, perc=1):
        self.max_features=max_Features
        self.perc = perc
        self.threshold = threshold
        self.xDecimals = xDecimals
        self.minLeafNodeSize = minLeafNodeSize
        self.bTree = []
        self.noOfTrees = noOfTrees
        for i in range(noOfTrees):
            # tempTree = bRegressionTree(threshold = self.threshold, xDecimals = self.xDecimals
            #                            , minLeafNodeSize=self.minLeafNodeSize)
            tempTree = bDecisionTree(threshold = self.threshold, xDecimals = self.xDecimals
                                       , minLeafNodeSize=self.minLeafNodeSize)
            self.bTree.append(tempTree)


    def fit(self,X,y):
        self.samples = []
        for i in range(self.noOfTrees):
            bootstrapSample = np.random.randint(X.shape[0],size=int(self.perc*X.shape[0]))
            self.samples.append(bootstrapSample)
            bootstrapX = X[bootstrapSample,:]
            bootstrapY = y[bootstrapSample]
            self.bTree[i].fit(bootstrapX,bootstrapY)



    def predict(self,X):
        y_predict = np.zeros(X.shape[0],self.noOfTrees)
        for i in range(self.noOfTrees):
            y_predict[:,1] =self.bTree[i].predict(X)
        result = np.unique(np.argmax(y_predict)) #Mit Unique noch die result rechnen.

        # ypredict = np.zeros(X.shape[0])
        # for i in range(self.noOfTrees):
        #     ypredict += self.bTree[i].predict(X)
        # ypredict = ypredict/self.noOfTrees
        return(result)

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


if __name__ == '__main__':
    dataset = np.loadtxt("iris.csv", delimiter=",")

    np.random.seed(42)
    MainSet = np.arange(0,dataset.shape[0])
    Trainingsset = np.random.choice(dataset.shape[0], 120, replace=False)
    Testset = np.delete(MainSet,Trainingsset)
    XTrain = dataset[Trainingsset,0:4]
    yTrain = dataset[Trainingsset,4]
    XTest = dataset[Testset,0:4]
    yTest = dataset[Testset,4]

    myTree = bDecisionTree(minLeafNodeSize=3)
    myTree.fit(XTrain,yTrain)

    yPredict = myTree.predict(XTest)
    acc = calculate_accuracy(yTest,yPredict)


    import  pandas as pd
    data = pd.read_csv('autos.csv')
    X=data.drop(columns='Fahrzeugklasse').select_dtypes('number')
    y = data['Fahrzeugklasse']
    x_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    autotree = bDecisionTree(minLeafNodeSize=3)
    autotree.fit(x_train,y_train)
    ypredauto = autotree.predict(X_test)
    calculate_accuracy(y_test,ypredauto)
