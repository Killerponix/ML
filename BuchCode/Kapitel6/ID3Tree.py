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

from binaryTree import tree 

class ID3BinaryTree:
    def __init__(self):
        self.bTree = None

    def _chooseFeature(self,X,y):
        # berechne die bedingte Entropie        
        H = np.zeros(X.shape[1])
        for i in range(len(H)):
            H[i] = CalConditionalEntropy(y,X,i)
        chooseA = np.argmin(H) # Waehle die kleinste bedingte Entropie aus
        return(chooseA)

    def _GenTree(self,X,y,parentNode,branch,A):
        if parentNode == None: # Wurzelknoten muss noch angelegt werden
            A = np.arange(X.shape[1])
        else:
            if len(y) == np.sum(y): # Nur noch positive Faelle vorhanden?
                    self.bTree.addNode(parentNode,branch,True)
                    return()
            elif 0 == np.sum(y):
                    self.bTree.addNode(parentNode,branch,False)
                    return()
            commonValue = True if np.sum(y)>len(y)/2 else False        
            if X.shape[0] == 0: # Keine Merkmale mehr vorhanden?
                self.bTree.addNode(parentNode,branch,commonValue)
                return()
        chooseA = self._chooseFeature(X,y) 
            
        if parentNode == None: # Wurzelknoten muss noch angelegt werden
            self.bTree = tree(chooseA, True, '=')
            myNo = 0
        else: # erzeuge neuen Knoten im Baum
            myNo = self.bTree.addNode(parentNode,branch,True, operator='=', varNo=A[chooseA])
        
        # loesche Merkmal in X in A
        index  = np.flatnonzero(np.logical_and(X[:,chooseA], 1))
        X = np.delete(X,chooseA,axis=1)
        A = np.delete(A,chooseA,axis=0)
        # teile X auf
        XTrue  = X[index,:]
        yTrue  = y[index]
        XFalse = np.delete(X,index,axis=0) 
        yFalse = np.delete(y,index,axis=0) 
        if XTrue.shape[0]>0: 
            self._GenTree(XTrue,yTrue,myNo,True,A)
        else:
            self.bTree.addNode(myNo,True,commonValue)
        if XFalse.shape[0]>0:
            self._GenTree(XFalse,yFalse,myNo,False,A)
        else:
            self.bTree.addNode(myNo,False,commonValue)
        return()

    def fit(self, X,y):
        self._GenTree(X,y,None,None,None)
    
    def predict(self, X):
        return(self.bTree.eval(X))
    
    def decisionPath(self, X):
        return(self.bTree.trace(X))
        
    def weightedPathLength(self,X):
        return(self.bTree.weightedPathLength(X)) 
        
    def numberOfLeafs(self):
        return(self.bTree.numberOfLeafs())
        
    


myTree = ID3BinaryTree()
myTree.fit(x,y)