import numpy as np 
from CARTRegressionTreeRF import bRegressionTree

class randomForestRegression:
    def __init__(self,noOfTrees=10,threshold = 10**-8, xDecimals = 8, minLeafNodeSize=3, perc=1):
        self.perc = perc
        self.threshold = threshold
        self.xDecimals = xDecimals
        self.minLeafNodeSize = minLeafNodeSize
        self.bTree = []
        self.noOfTrees = noOfTrees
        for i in range(noOfTrees):
            tempTree = bRegressionTree(threshold = self.threshold, xDecimals = self.xDecimals , minLeafNodeSize=self.minLeafNodeSize)
            self.bTree.append(tempTree)
            
    def fit(self,X,y):
        self.samples = []
        for i in range(self.noOfTrees):
            bootstrapSample = np.random.randint(X.shape[0],size=int(self.perc*X.shape[0]))
            self.samples.append(bootstrapSample)     #*\label{code:realRF:0}
            bootstrapX = X[bootstrapSample,:]
            bootstrapY = y[bootstrapSample]
            self.bTree[i].fit(bootstrapX,bootstrapY)
    
    def predict(self,X):
        ypredict = np.zeros(X.shape[0])
        for i in range(self.noOfTrees):
            ypredict += self.bTree[i].predict(X)
        ypredict = ypredict/self.noOfTrees
        return(ypredict)
        
if __name__ == '__main__':   
    f = open("hourCleanUp.csv") #*\label{code:realRF:1}
    header = f.readline().rstrip('\n')  
    featureNames = header.split(',')
    dataset = np.loadtxt(f, delimiter=",")
    f.close()
    
    X = dataset[:,0:13]
    Y = dataset[:,15]
    
    index = np.flatnonzero(X[:,8]==4)
    X = np.delete(X,index, axis=0)
    Y = np.delete(Y,index, axis=0)
    
    np.random.seed(42)
    MainSet = np.arange(0,X.shape[0])
    Trainingsset = np.random.choice(X.shape[0], int(0.8*X.shape[0]), replace=False)
    Testset = np.delete(MainSet,Trainingsset)
    XTrain = X[Trainingsset,:]
    yTrain = Y[Trainingsset]
    XTest = X[Testset,:]
    yTest = Y[Testset] #*\label{code:realRF:2}
    
    myForest = randomForestRegression(noOfTrees=24,minLeafNodeSize=5,threshold=2)
    myForest.fit(XTrain,yTrain)
    yPredict = np.round(myForest.predict(XTest))
    yDiff = yPredict - yTest
    print('Mittlere Abweichung: %e ' % (np.mean(np.abs(yDiff))))
                