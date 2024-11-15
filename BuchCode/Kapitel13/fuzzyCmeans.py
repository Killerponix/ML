import numpy as np

class fuzzyCmeans:
    def __init__(self, noOfClusters ,p=2): 
        self.noOfClusters = noOfClusters
        self.p = p 
        self.fitHistory = []
        
    def _computeDistances(self,X,centres):
        distances = ( np.sum( np.abs(X-centres[0,:])**self.p,axis=1) )**(1/self.p)
        for j in range(1,self.noOfClusters):
            distancesAdd = ( np.sum( np.abs(X-centres[j,:])**self.p,axis=1) )**(1/self.p)
            distances = np.vstack( (distances,distancesAdd) )
        return(distances)
        
    def fit(self, X, maxIterations=42, vareps=10**-3):
        Xmin = X.min(axis=0)
        Xmax = X.max(axis=0)
        newcentres = np.random.rand(self.noOfClusters,X.shape[1])*(Xmax - Xmin)+Xmin
        oldCentres = newcentres + 1
        count = 0
        while np.sum(np.abs(oldCentres-newcentres)) > vareps and count<maxIterations:
            count = count + 1
            oldCentres = np.copy(newcentres)
            self.fitHistory.append(newcentres.copy())
            
            distances =self._computeDistances(X,newcentres) #*\label{code:fuzzyCmeans:0}  
            d2 = distances**2 #*\label{code:fuzzyCmeans:1} 
            d2Sum = np.sum(1/d2,axis=0) #*\label{code:fuzzyCmeans:2}
            W = d2*d2Sum #*\label{code:fuzzyCmeans:3}
            W = 1/W #*\label{code:fuzzyCmeans:4}
            WSum = np.sum(W**2,axis=1) #*\label{code:fuzzyCmeans:5}
            omega = ((W**2).T/WSum).T #*\label{code:fuzzyCmeans:6}
            newcentres = omega@X
        
        self.fitHistory.append(newcentres.copy())
        self.centers = newcentres
        return(newcentres,W.T)
        
    def predict(self,X):
        distances =self._computeDistances(X,self.centres)
        cluster = distances.argmin(axis=0)
        return cluster
    
if __name__ == '__main__':
    def bubbleSetNormal(mx,my,number,s):
        x = np.random.normal(0, s, number) + mx
        y = np.random.normal(0, s, number) + my
        return(x,y)
    
    def mouseShape():
        np.random.seed(42)        
        dataset = np.zeros( (1000,2) )    
        (dataset[0:150,0],dataset[0:150,1])     = bubbleSetNormal(-0.75, 0.75,150,0.15)     
        (dataset[150:300,0],dataset[150:300,1]) = bubbleSetNormal( 0.75, 0.75,150,0.15)    
        (dataset[300:1000,0],dataset[300:1000,1]) = bubbleSetNormal( 0, 0,700,0.29)   
        return (dataset) 
    # Testbeispiel 3
    X = mouseShape()
    noOfClusters  = 3 
    cAlgo = fuzzyCmeans(noOfClusters)
    newcentres, W = cAlgo.fit(X)
    
    cluster = np.argmax(W,axis=1)
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure()   
    ax = fig.add_subplot(1,1,1)
    ax.scatter(X[:,0]  ,X[:,1],c=W[:,0],s=60, cmap='binary')  
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True,linestyle='-',color='0.75')
    ax.set_aspect('equal', 'datalim')
    
    fig = plt.figure()   
    ax = fig.add_subplot(1,1,1)
    ax.scatter(X[:,0]  ,X[:,1],c=W[:,1],s=60, cmap='binary')  
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True,linestyle='-',color='0.75')
    ax.set_aspect('equal', 'datalim')
    
    fig = plt.figure()   
    ax = fig.add_subplot(1,1,1)
    ax.scatter(X[:,0]  ,X[:,1],c=W[:,2],s=60, cmap='binary')  
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True,linestyle='-',color='0.75')
    ax.set_aspect('equal', 'datalim')
    
    