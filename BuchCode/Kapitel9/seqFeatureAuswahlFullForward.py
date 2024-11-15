import numpy as np
from CARTRegressionTree import bRegressionTree

def SFS(X,y,k, verbose=False):
    MainSet = np.arange(0,X.shape[0])
    ValSet  = np.random.choice(X.shape[0],int(X.shape[0]*0.25), replace=False)
    TrainSet  = np.delete(MainSet,ValSet)    
    featuresLeft = np.arange(0,X.shape[1]) 
    suggestedFeatures = np.zeros(1,dtype=int)
    l=0
    while (k>l):
        Q = np.inf*np.ones(X.shape[1])
        for i in featuresLeft:
            suggestedFeatures[l] = i
            reduTree = bRegressionTree(minLeafNodeSize=40)
            reduTree.fit(X[np.ix_(TrainSet,suggestedFeatures)],y[TrainSet])
            error = y[ValSet] - reduTree.predict(X[np.ix_(ValSet,suggestedFeatures)])
            Q[i] = np.mean(np.abs(error)) 
        i = np.argmin(Q)
        if verbose: print(Q);print(i)
        suggestedFeatures[l] = i
        featuresLeft = np.delete(featuresLeft,np.argwhere(featuresLeft == i) ) 
        suggestedFeatures = np.hstack( (suggestedFeatures,np.array([0]) ) )
        l = l +1
    suggestedFeatures = suggestedFeatures[0:l]
    return(suggestedFeatures)

np.random.seed(999)

X = np.random.rand(1000,5) #*\label{seqFeature:26}
y = 2*X[:,1] - X[:,3]**2 - 0.01*X[:,0]**3 + 0.1*(X[:,2] - X[:,4]**2) #*\label{seqFeature:27}

suggestedFeatures = SFS(X,y,2, verbose=True)
print(suggestedFeatures)
