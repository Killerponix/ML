import numpy as np
from CARTRegressionTree import bRegressionTree

def SBS(X,y,k, verbose=False):
    l=X.shape[1]
    MainSet = np.arange(0,X.shape[0])
    ValSet  = np.random.choice(X.shape[0],int(X.shape[0]*0.25), replace=False)
    TrainSet  = np.delete(MainSet,ValSet)    
    suggestedFeatures = np.arange(0,l) #*\label{seqFeature:9}
    while (k<l):
        Q = np.zeros(l)
        for i in range(l):
            Xred = np.delete(X, i, axis=1)
            reduTree = bRegressionTree(minLeafNodeSize=40)
            reduTree.fit(Xred[TrainSet,:],y[TrainSet])
            error = y[ValSet] - reduTree.predict(Xred[ValSet,:])
            Q[i] = np.mean(np.abs(error)) #*\label{seqFeature:17}
        i = np.argmin(Q)
        if verbose: print(Q);print(suggestedFeatures[i])
        suggestedFeatures = np.delete(suggestedFeatures,i) #*\label{seqFeature:19}
        X = np.delete(X, i, axis=1)
        l = l -1
    return(suggestedFeatures)

np.random.seed(42)

X = np.random.rand(1000,5) #*\label{seqFeature:26}
y = 2*X[:,1] - X[:,3]**2 - 0.01*X[:,0]**3 + 0.1*(X[:,2] - X[:,4]**2) #*\label{seqFeature:27}

suggestedFeatures = SBS(X,y,2, verbose=True)
print(suggestedFeatures)
