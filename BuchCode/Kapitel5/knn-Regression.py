import numpy as np
from scipy.spatial import KDTree

class knnRegression:
    def fit(self,X,Y):
        self.xMin = X.min(axis=0)
        self.xMax = X.max(axis=0) 
        self.XTrain = (X - self.xMin) / (self.xMax - self.xMin)    
        self.kdTree = KDTree(self.XTrain) 
        self.YTrain = Y
    
    def predict(self,X, k=3, smear = 1):
        X = (X - self.xMin) / (self.xMax - self.xMin)
        (dist, neighbours) = self.kdTree.query(X,k)
        distsum = np.sum( 1/(dist+smear/k), axis=1) #*\label{code:knnReg1}
        distsum = np.repeat(distsum[:,None],k,axis=1) 
        dist = (1/distsum)*1/(dist + smear/k) #*\label{code:knnReg2}
        y = np.sum( dist*self.YTrain[neighbours],axis=1)
        return(y)

if __name__ == '__main__':
    samples = 5000 
    pNoise = 1
    myK = 3
    mysmear = 0.5
    
    np.random.seed(42)
    x = np.random.rand(samples,2)
    y = np.tanh( 500*( (1/16) - (x[:,0]-0.5)**2 - (x[:,1]-0.5)**2 ) )
    Noise = np.random.normal(size=len(y))
    y = (1+Noise*pNoise/100)*y
    
    percentTrainingset = 0.8
    TrainSet     = np.random.choice(x.shape[0],int(x.shape[0]*percentTrainingset),replace=False)
    XTrain       = x[TrainSet,:]
    YTrain       = y[TrainSet]
    TestSet      = np.delete(np.arange(0,len(y)), TrainSet) 
    XTest        = x[TestSet,:]
    YTest        = y[TestSet]
    
    myRegression = knnRegression()
    myRegression.fit(XTrain,YTrain)
    yP = myRegression.predict(XTest,k=myK, smear=mysmear)
    diff = yP-YTest
    MAE = np.mean(np.abs(diff))
    print(MAE)
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter(XTest[:,0],XTest[:,1],yP,alpha=0.6,c =yP, cmap=cm.copper)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$y_P$')
    

#
#
#lookUpTree = KDTree(x) 
#closeToCenter = lookUpTree.query_ball_point([0, 0], 0.2)
#print(closeToCenter)
#print(x[closeToCenter,:])