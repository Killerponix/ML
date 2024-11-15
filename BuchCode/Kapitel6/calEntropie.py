import numpy as np

def weightedSelfInformation( x ):
    y = 0 if x <= 0 else x*np.log2(x)
    return(y)

def CalConditionalEntropy(y,D,Feature):
    sizeDataBase = D.shape[0] 
    D = D.astype(bool)
    TrueFeatureDatabase  = np.sum(D[:,Feature])
    FalseFeatureDatabase = sizeDataBase - TrueFeatureDatabase
    PFeatureTrue  = TrueFeatureDatabase/sizeDataBase
    PFeatureFalse = FalseFeatureDatabase/sizeDataBase
    
    Htrue = 0
    if PFeatureTrue>0:
        P_AB_True  = TrueFeatureDatabase - np.sum(np.logical_and(D[:,Feature],y))
        P_AB_False = TrueFeatureDatabase - P_AB_True
        P_AB_True  = P_AB_True/TrueFeatureDatabase
        P_AB_False = P_AB_False/TrueFeatureDatabase
        Htrue      = PFeatureTrue * (weightedSelfInformation(P_AB_False) + weightedSelfInformation(P_AB_True) )
    Hfalse = 0
    if PFeatureFalse>0:
        P_AB_True  = FalseFeatureDatabase - np.sum(np.logical_and(~D[:,Feature],y))
        P_AB_False = FalseFeatureDatabase - P_AB_True
        P_AB_True  = P_AB_True/FalseFeatureDatabase
        P_AB_False = P_AB_False/FalseFeatureDatabase
        Hfalse     = PFeatureFalse * (weightedSelfInformation(P_AB_False) + weightedSelfInformation(P_AB_True) )
    
    H = -Htrue - Hfalse
    return(H)  
    
dataSet = np.array([[ 1  , 0   ,  0  , 0  , 1 ] , [  0  , 0   ,  0  , 0  , 0 ] , 
                    [ 0  , 0   ,  1  , 0  , 1 ] , [  0  , 0   ,  0  , 0  , 0 ] , 
                    [ 1  , 0   ,  1  , 0  , 1 ] , [  0  , 0   ,  1  , 1  , 1 ] , 
                    [ 1  , 0   ,  0  , 1  , 0 ] , [  1  , 1   ,  0  , 0  , 0 ] , 
                    [ 1  , 1   ,  1  , 0  , 0 ] ])
x = dataSet[:,0:4]
y = dataSet[:,4]

for i in range(4):
    H = CalConditionalEntropy(y,x,i)
    print(H)


   