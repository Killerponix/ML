import numpy as np
import scipy.special 
import copy

class simpleMLP:           
    def __init__(self, hiddenlayer=(10,10)):
        self.hl = hiddenlayer
        self.xMin = 0.0; self.xMax = 1.0
        self.W = []
        self._sigmoid = lambda x: scipy.special.expit(x)

    def _initWeights(self):
        self.W.append((np.random.rand(self.hl[0],self.il) - 0.5 ))
        self.W.append((np.random.rand(self.hl[1],self.hl[0]) - 0.5))
        self.W.append((np.random.rand(self.ol,self.hl[1]) - 0.5))

    def _calOut(self,X):
        O1 = self._sigmoid(self.W[0]@X.T)
        O2 = self._sigmoid(self.W[1]@O1)
        y = (self.W[len(self.W)-1]@O2).T
        return(y)
    
    def predict(self,X):
        X = (X - self.xMin) / (self.xMax - self.xMin) #*\label{code:mlpbatch:predict:norm}
        X = np.hstack( (X,np.ones(X.shape[0])[:,None]) ) #*\label{code:mlpbatch:bias1}
        y = self._calOut(X)
        
        return(y)

    def fit(self,X,Y,eta=0.75,maxIter=200,vareps=10**-3,scale=True):
        self.xMin = X.min(axis=0) if scale else 0
        self.xMax = X.max(axis=0) if scale else 1
        X = (X - self.xMin) / (self.xMax - self.xMin)
        X = np.hstack( (X,np.ones(X.shape[0])[:,None]) ) #*\label{code:mlpbatch:bias2}
        if len(Y.shape) == 1:
            Y = Y[:,None]
        self.il = X.shape[1] #*\label{code:mlpbatch:initw1}
        self.ol = 1
        self._initWeights() #*\label{code:mlpbatch:initw2}
        
        self.train(X,Y,eta,maxIter,vareps)
     
    def train(self,X,Y,eta,maxIter=200,vareps=10**-3): 
        if len(Y.shape) == 1: Y = Y[:,None]
        
        if self.il != X.shape[1]: X = np.hstack( (X,np.ones(X.shape[0])[:,None]) )

        
        dW = [] #*\label{code:mlpbatch:initdw:s}
        for i in range(len(self.W)):
            dW.append(np.zeros_like(self.W[i])) #*\label{code:mlpbatch:initdw:e}
        yp = self._calOut(X)
        
        meanE = np.sum((Y-yp)**2)/X.shape[0] #*\label{code:mlp:error:1}
        minError = meanE #*\label{code:mlp:error:2}
        minW = copy.deepcopy(self.W) #*\label{code:mlp:error:3} 
        self.error=[]   #*\label{code:mlp:error:4}   
        mixSet = np.random.choice(X.shape[0],X.shape[0],replace=False) #*\label{code:mlp:mischen}
        counter = 0            
        while meanE > vareps and counter < maxIter: #*\label{code:mlpbatch:abbruch}
            counter += 1
            
            for i in mixSet:  #*\label{code:mlpbatch:-1}         
                x = X[i,:]
                O1 = self._sigmoid(self.W[0]@x.T) 
                O2 = self._sigmoid(self.W[1]@O1)
                temp = self.W[2]*O2*(1-O2)[None,:]
                dW[2] = O2 #*\label{code:mlpbatch:2}
                dW[1] = temp.T@O1[:,None].T #*\label{code:mlpbatch:3}   
                dW[0] = (O1*(1-O1)*(temp@self.W[1])).T@x[:,None].T #*\label{code:mlpbatch:4}  
                yp = self._calOut(x)
                yfactor = np.sum(Y[i]-yp)
                for j in range(len(self.W)): #*\label{code:mlpbatch:update}
                    self.W[j] += eta * yfactor* dW[j]
    
            yp = self._calOut(X)      #*\label{code:mlperror:1}     
            
            meanE = (np.sum((Y-yp)**2)/X.shape[0])
            self.error.append(meanE)
            if meanE < minError: 
                minError = meanE
                minW = copy.deepcopy(self.W)  #*\label{code:mlperror:2}    
        self.W = copy.deepcopy(minW) #*\label{code:mllcopy}
        
if __name__ == '__main__':
    np.random.seed(42)
    XTrain = np.random.rand(2500,2)
    YTrain = np.sin(2*np.pi*(XTrain[:,0] + 0.5*XTrain[:,1])) + 0.5*XTrain[:,1] 
    Noise = np.random.rand(YTrain.shape[0]) - 0.5 
    YTrain = (1+ 0.05*Noise)*YTrain 
    
    XTest = np.random.rand(500,2)
    YTest = np.sin(2*np.pi*(XTest[:,0] + 0.5*XTest[:,1])) + 0.5*XTest[:,1]
    
    myPredict = simpleMLP(hiddenlayer=(8,8))
    myPredict.fit(XTrain,YTrain)
    yp = np.squeeze(myPredict.predict(XTest))
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(1,1,1, projection='3d')
    ax.scatter(XTest[:,0],XTest[:,1],yp,alpha=0.6,c =yp, cmap=cm.Greys)
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    ax.set_zlabel('$y_p$')
    
    fig2 = plt.figure(2)
    ax = fig2.add_subplot(1,1,1, projection='3d')
    ax.scatter(XTest[:,0],XTest[:,1],yp.T-YTest,alpha=0.6,c =yp.T-YTest, cmap=cm.Greys)
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    ax.set_zlabel('$y_p - y$')
    
    fig3 = plt.figure(3)
    ax = fig3.add_subplot(1,1,1)
    epochen = np.arange(len(myPredict.error))
    ax.plot(epochen, np.array(myPredict.error), 'k' )  
    ax.set_xlabel('Lernzyklus')
    ax.set_ylabel('Durchschnittlicher Fehler')