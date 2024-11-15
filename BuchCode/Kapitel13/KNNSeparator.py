import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans

class knnRegressionMV:
    def fit(self, X, Y, Quality = np.NaN, takeAddOnFactor = 2.25, QMultiplicator = 2):
        if X.shape[0] > 200000: lsize = 200
        else: lsize = 10
        self.kdTree = KDTree(X, leaf_size=lsize, metric='minkowski', p=2)
        self.YTrain = Y
        self.Quality = Quality
        self.takeAddOnFactor = takeAddOnFactor
        self.QMultiplicator  = QMultiplicator
    
    def predict(self,X, kFin=3, smear = 1,answers=2):
        sizeLocalGroup = int(kFin*answers*self.takeAddOnFactor)
        if np.any(np.isnan(self.Quality)): k = sizeLocalGroup
        else: k = int(kFin*answers*self.takeAddOnFactor*self.QMultiplicator)
        (distInit, neighboursInit) = self.kdTree.query(X,k)
        
        if ~np.any(np.isnan(self.Quality)):
            neighbours = np.zeros( (neighboursInit.shape[0],sizeLocalGroup), dtype=int )
            dist = np.zeros( (distInit.shape[0],sizeLocalGroup), dtype=float )
            for i in range(neighboursInit.shape[0]):
                localQuality = self.Quality[neighboursInit[i,:]]
                choose = np.argsort(localQuality)[0:sizeLocalGroup]
                neighbours[i,:] = neighboursInit[i,choose]
                dist[i,:] = distInit[i,choose]
            YNeighbours = np.zeros( (self.YTrain[neighbours].shape[0],
                                     self.YTrain[neighbours].shape[1],
                                     self.YTrain[neighbours].shape[2]+1) )
            YNeighbours[:,:,0:self.YTrain[neighbours].shape[2]] = self.YTrain[neighbours]
            YNeighbours[:,:,self.YTrain[neighbours].shape[2]]  = self.Quality[neighbours]
        else:
            dist = distInit
            neighbours = neighboursInit
            YNeighbours = self.YTrain[neighbours]
        
        YNeighboursReduced= np.zeros((answers,YNeighbours.shape[0],kFin,YNeighbours.shape[2]))
        distReduced = np.zeros( (answers,dist.shape[0],kFin) )
        
        for i in range(YNeighbours.shape[0]): 
            YN = YNeighbours[i,:]
            kmeans = KMeans(n_clusters=answers, random_state=0).fit(YN)
            for antwort in range(answers): 
                index = np.flatnonzero(kmeans.labels_ == antwort)
                if len(index) >= kFin:
                    YNA = YN[index]   
                    dYN =  distance_matrix(YNA,YNA)
                    choose = np.argsort(dYN,axis=1)[:,0:kFin]
                    temp = np.zeros((choose.shape[0],kFin))
                    for j in range(dYN.shape[0]):
                        temp[j,:] = dYN[j,choose[j,:]]
                    dk = np.sum(temp,axis=1)
                    theChoosenOnce = np.argsort(dk)[0]   
                    globalIndex = index[choose[theChoosenOnce]]
                    YNeighboursReduced[antwort,i,:] = YNA[choose[theChoosenOnce]]
                    distReduced[antwort,i,:] = dist[i,globalIndex]
                else: 
                    YNeighboursReduced[antwort,i,:] = np.NaN 
                    distReduced[antwort,i,:] = np.NaN 
        
        yAll = np.zeros( (answers,X.shape[0],YNeighbours.shape[2]) )
        for antwort in range(answers):    
            distReducedA = distReduced[antwort,:]
            YNeighboursReducedA = YNeighboursReduced[antwort,:]
            distsum = np.sum( 1/(distReducedA+smear/kFin), axis=1) 
            distsum = np.repeat(distsum[:,None],kFin,axis=1) 
            dist = (1/distsum)*1/(distReducedA + smear/kFin) 
            y = np.zeros((X.shape[0],YNeighbours.shape[2]))
            for i in range(YNeighbours.shape[2]):
                y[:,i] = np.sum( dist*YNeighboursReducedA[:,:,i],axis=1)
            yAll[antwort,:] = y
        yAll = np.swapaxes(yAll,0,1)
        return yAll

if __name__ == '__main__':   
    import pandas as pd  
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    dataset = np.unique(pd.read_csv("xorfile.csv",delimiter=",").to_numpy(),axis=0)
    np.random.seed(42)
    idx = np.random.choice(np.arange(dataset.shape[0]), 1000,replace=False)
    X = dataset[idx,0:3]
    Y = dataset[idx,3:12]
    Q = dataset[idx,12]

    mySep = knnRegressionMV()
    mySep.fit(X, Y, Quality=Q, QMultiplicator  = 1.25, takeAddOnFactor = 2.0)
    
    errorQ = []; errorC = []; lossValue = []
    solutionsFound = 0
    for tests in range(100):
        b = np.array( [ ( np.random.rand())/2 , ( np.random.rand())/2] ) # XOR-Offset
        alpha = 2*(np.random.rand()-0.5) * np.pi*(45/180) # XOR-Winkel
        A = np.zeros((2,2))
        A[0,0] = np.cos(alpha); A[1,1] =   A[0,0]
        A[0,1] = np.sin(alpha); A[1,0] = - A[0,1]
        
        k = 10000  # number of samples of each class
        X00 = 0.25*(np.random.rand(int(k/4),2)) + A@([0.00,0.00] + b) 
        X11 = 0.25*(np.random.rand(int(k/4),2)) + A@([0.75,0.75] + b) 
        X10 = 0.25*(np.random.rand(int(k/4),2)) + A@([0.75,0.00] + b) 
        X01 = 0.25*(np.random.rand(int(k/4),2)) + A@([0.00,0.75] + b) 
        XTest = np.vstack( (X00,X11,X10,X01))
        YTest  = np.hstack( (np.zeros(X00.shape[0]),np.zeros(X11.shape[0]),
                             np.ones(X10.shape[0]),np.ones(X01.shape[0]) ) )
        
        x = np.array([b[0],b[1],alpha]).reshape(1,3) 
        y = mySep.predict(x, kFin=5, answers=6, smear = 0.1)
        
        for i in range(y.shape[1]):
            if np.any(np.isnan(y[0,i,0:4])): continue
            solutionsFound += 1
    
            myANN = Sequential()
            myANN.add(Dense(2,input_dim=2, activation='sigmoid',use_bias=True))
            myANN.add(Dense(1,activation='sigmoid',use_bias=True))
            myANN.compile(loss='binary_crossentropy', optimizer='SGD',  metrics=['acc'])
            
            B0 = y[0,i,0:4].reshape(2,2)
            b0 = y[0,i,4:6].reshape(2,)
            B1 = y[0,i,6:8].reshape(2,1)
            b1 = y[0,i,8].reshape(1,)
            WStart0 = [B0,b0]
            WStart1 = [B1,b1]      
            myANN.layers[0].set_weights(WStart0)
            myANN.layers[1].set_weights(WStart1)
            
            quality = y[0,i,9]
            q, correct = myANN.evaluate(XTest,YTest)
            errorQ.append(quality-q)
            errorC.append(1-correct)
            lossValue.append(q)
    print(np.mean(errorQ), np.mean(errorC), np.mean(lossValue))
