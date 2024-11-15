import numpy as np
import scipy.special 
import copy

class MLPNet:            
    def __init__(self, hiddenlayer=(10,10),classification=False):
        self.hl = hiddenlayer; self.classification = classification
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
        X = (X - self.xMin) / (self.xMax - self.xMin)
        X = np.hstack( (X,np.ones(X.shape[0])[:,None]) ) 
        y = self._calOut(X)
        if self.classification: y = np.round(y) #*\label{code:fullmlpbatch:1}
        return(y)
    
    def fit(self,X,Y,eta=0.75,maxIter=200,vareps=10**-3,scale=True,XT=None,YT=None): #*\label{code:fullmlpbatch:4}
        self.xMin = X.min(axis=0) if scale else 0
        self.xMax = X.max(axis=0) if scale else 1
        X = (X - self.xMin) / (self.xMax - self.xMin)
        X = np.hstack( (X,np.ones(X.shape[0])[:,None]) ) 
        if len(Y.shape) == 1:
            Y = Y[:,None] 
        self.il = X.shape[1] 
        self.ol = Y.shape[1] #*\label{code:fullmlpbatch:morethanone}
        self._initWeights()
        (XVal, YVal, X, Y) = self._divValTrainSet(X,Y)         #*\label{code:fullmlpbatch:3}
        self.train(X,Y,XVal,YVal,eta,maxIter,vareps,XT,YT)
     
    def train(self,X,Y,XVal=None,YVal=None,eta=0.75,maxIter=200,vareps=10**-3,XT=None,YT=None):    
        if XVal is None: (XVal, YVal, X, Y) = self._divValTrainSet(X,Y) 
        if len(Y.shape) == 1: Y = Y[:,None]
        if len(YVal.shape) == 1: YVal = YVal[:,None]
        if self.il != X.shape[1]: X = np.hstack( (X,np.ones(X.shape[0])[:,None]) )
        if self.il != XVal.shape[1]: XVal = np.hstack( (XVal,np.ones(XVal.shape[0])[:,None]) )
        dW = []
        for i in range(len(self.W)):
            dW.append(np.zeros_like(self.W[i])) 
        yp = self._calOut(XVal)
        if self.classification: yp = np.round(yp)
        meanE = (np.sum((YVal-yp)**2)/XVal.shape[0])/YVal.shape[1]
        minError = meanE
        minW = copy.deepcopy(self.W) 
        self.errorVal=[]; self.errorTrain=[]; self.errorTest=[] #*\label{code:fullmlpbatch:2}
        mixSet = np.random.choice(X.shape[0],X.shape[0],replace=False)
        counter = 0            
        while meanE > vareps and counter < maxIter: 
            counter += 1
            for m in range(self.ol): #*\label{code:fullmlpbatch:5}
                for i in mixSet: 
                    x = X[i,:]
                    O1 = self._sigmoid(self.W[0]@x.T) 
                    O2 = self._sigmoid(self.W[1]@O1) 
                    temp = self.W[2][m,:]*O2*(1-O2)[None,:]
                    dW[2] = O2 
                    dW[1] = temp.T@O1[:,None].T   
                    dW[0] = (O1*(1-O1)*(temp@self.W[1])).T@x[:,None].T 
                    yp = self._calOut(x)[m]
                    yfactor = np.sum(Y[i,m]-yp)
                    for j in range(len(self.W)):     
                        self.W[j] += eta * yfactor* dW[j] 

            yp = self._calOut(XVal)
            if self.classification: yp = np.round(yp)
            meanE = (np.sum((YVal-yp)**2)/XVal.shape[0])/YVal.shape[1] #*\label{code:fullmlpbatch:6}
            self.errorVal.append(meanE)
            if meanE < minError: 
                minError = meanE
                minW = copy.deepcopy(self.W)      
                self.valChoise = counter
                
            if XT is not None:
                yp = self.predict(XT)
                if len(YT.shape) == 1: YT = YT[:,None]; 
                meanETest = (np.sum((YT-yp)**2)/XT.shape[0])/YT.shape[1]
                self.errorTest.append(meanETest)
                
                yp = self._calOut(X)
                if self.classification:
                    yp = np.round(yp)
                meanETrain = (np.sum((Y-yp)**2)/X.shape[0])/Y.shape[1]
                self.errorTrain.append(meanETrain)
        self.W = copy.deepcopy(minW) 
    
    def _divValTrainSet(self, X,Y):
        self.ValSet    = np.random.choice(X.shape[0],int(X.shape[0]*0.25),replace=False)
        self.TrainSet  = np.delete(np.arange(0, Y.shape[0] ), self.ValSet) 
        XVal     = X[self.ValSet,:]
        YVal     = Y[self.ValSet]
        X        = X[self.TrainSet,:]
        Y        = Y[self.TrainSet]
        return (XVal, YVal, X, Y)
    
    def exportNet(self, filePrefix):
        np.savetxt(filePrefix+"MinMax.csv", np.array([self.xMin, self.xMax]), delimiter=",")
        np.savetxt(filePrefix+"W0.csv", self.W[0], delimiter=",")
        np.savetxt(filePrefix+"W1.csv", self.W[1], delimiter=",")
        np.savetxt(filePrefix+"W2.csv", self.W[2], delimiter=",")
    
    def importNet(self,filePrefix, classification=False):
        MinMax = np.loadtxt(filePrefix+'MinMax.csv',delimiter=",")
        W2 = np.loadtxt(filePrefix+'W2.csv',delimiter=",")
        W1 = np.loadtxt(filePrefix+'W1.csv',delimiter=",")    
        W0 = np.loadtxt(filePrefix+'W0.csv',delimiter=",") 
        self.W = [W0,W1,W2]
        self.hl = (W0.shape[0], W2.shape[1])
        self.il = W0.shape[1]
        self.ol = W2.shape[0]
        self.xMin = MinMax[0] 
        self.xMax = MinMax[1]
        self.classification = classification

if __name__ == '__main__':
    np.random.seed(42)
    X = np.random.rand(1250,2)
    Y = np.zeros( (1250,2) )
    index1 = (X[:,0] - 0.25)**2 + (X[:,1] - 0.25)**2 < 0.2**2
    Y[index1,0] = 1
    index2 = (X[:,0] - 0.75)**2 + (X[:,1] - 0.75)**2 < 0.2**2
    Y[index2,1] = 1
    
    TrainSet     = np.random.choice(X.shape[0],int(X.shape[0]*0.70), replace=False)
    XTrain       = X[TrainSet,:]
    YTrain       = Y[TrainSet]
    TestSet      = np.delete(np.arange(0, len(Y) ), TrainSet) 
    XTest        = X[TestSet,:]
    YTest        = Y[TestSet]
    
    myPredict = MLPNet(hiddenlayer=(24,24),classification=True)
    myPredict.fit(XTrain,YTrain,maxIter=1200, XT=XTest , YT=YTest)
    yp = myPredict.predict(XTest)
    
    fp = np.sum(np.abs(yp - YTest))/len(TestSet)*100 
    print('richtig klassifiziert %0.1f%%' % (100-fp))
    print('falsch klassifiziert %0.1f%%' % (fp))

    myPredict.exportNet('foobar')
    justTest = MLPNet()
    justTest.importNet('foobar',classification=True)
    yp = justTest.predict(XTest)
    fp = np.sum(np.abs(yp - YTest))/len(TestSet)*100 
    print('richtig klassifiziert %0.1f%%' % (100-fp))
    print('falsch klassifiziert %0.1f%%' % (fp))
    
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    plt.close('all')
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(1,1,1)
    circle1 = plt.Circle((0.25, 0.25), 0.2, color='k', alpha=0.3)
    circle2 = plt.Circle((0.75, 0.75), 0.2, color='k', alpha=0.3)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    
    index1 = np.logical_and( (XTest[:,0] - 0.25)**2 + (XTest[:,1] - 0.25)**2 < 0.2**2 , yp[: ,0]==0 )
    ax.scatter(XTest[index1,0],XTest[index1,1], marker='v',c='r')
    index2 = np.logical_and(  (XTest[:,0] - 0.75)**2 + (XTest[:,1] - 0.75)**2 < 0.2**2, yp[: ,1]==0 )
    ax.scatter(XTest[index2,0],XTest[index2,1], marker='^',c='r')
    
    ax.scatter(XTest[yp[:,0]==1,0],XTest[yp[:,0]==1,1], marker='+',c='k')
    ax.scatter(XTest[yp[:,1]==1,0],XTest[yp[:,1]==1,1], marker='o',c='k')
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis('equal')
    
    fig3 = plt.figure(3)
    ax = fig3.add_subplot(1,1,1)
    epochen = np.arange(len(myPredict.errorVal))
    ax.plot(epochen, np.array(myPredict.errorVal), 'r-.' , label='Validierung')  
    ax.plot(epochen, np.array(myPredict.errorTest), 'k--', label='Test')   
    ax.plot(epochen, np.array(myPredict.errorTrain), 'k:', label='Training' )  
    ax.legend()
    ax.set_xlabel('Lernzyklus')
    ax.set_ylabel('Durchschnittlicher Fehler')