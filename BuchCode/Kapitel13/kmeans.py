import numpy as np

class kmeans:
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
        
    def fit(self, X, maxIterations=42):
        Xmin = X.min(axis=0)
        Xmax = X.max(axis=0)
        newcentres = np.random.rand(self.noOfClusters,X.shape[1])*(Xmax - Xmin)+Xmin
        oldCentres = newcentres + 1
        count = 0
        while np.sum(np.abs(oldCentres-newcentres))!= 0 and count<maxIterations:
            count = count + 1
            oldCentres = np.copy(newcentres)
            self.fitHistory.append(newcentres.copy())
            distances =self._computeDistances(X,newcentres)
            cluster = distances.argmin(axis=0)
    
            for j in range(self.noOfClusters):
                index = np.flatnonzero(cluster == j)    
                if index.shape[0]>0:
                    newcentres[j,:] = np.sum(X[index,:],axis=0)/index.shape[0]
                else:
                    distances = ( np.sum( (X-newcentres[j,:])**self.p,axis=1) )**(1/self.p)
                    i = distances.argmin(axis=0)
                    newcentres[j,:] = X[i,:] 
                    cluster[i]=j
        self.centers = newcentres
        return(newcentres,cluster)
        
    def predict(self,X):
        distances =self._computeDistances(X,self.centres)
        cluster = distances.argmin(axis=0)
        return cluster
    
if __name__ == '__main__':
    def bubbleSetNormal(mx,my,number,s):
        x = np.random.normal(0, s, number) + mx
        y = np.random.normal(0, s, number) + my
        return(x,y)

    def fourBalls(n1,n2,n3,n4):
        np.random.seed(42)        
        dataset = np.zeros( (n1+n2+n3+n4,2) )    
        (dataset[0:n1,0],dataset[0:n1,1])     = bubbleSetNormal( 2.5, 1.0,n1,0.5)     
        (dataset[n1:n1+n2,0],dataset[n1:n1+n2,1]) = bubbleSetNormal( 2.0,-3.0,n2,0.3)    
        (dataset[n1+n2:n1+n2+n3,0],dataset[n1+n2:n1+n2+n3,1]) = bubbleSetNormal(-2.0, 5.0,n3,0.6)   
        (dataset[n1+n2+n3:n1+n2+n3+n4,0],dataset[n1+n2+n3:n1+n2+n3+n4,1]) = bubbleSetNormal(-4.0,-1.0,n4,0.9)        
        return (dataset)    
    
    n1=n2=n3=n4=400
    # Testbeispiel 1
    X = fourBalls(n1,n2,n3,n4)
    
    noOfClusters  = 4
    cAlgo = kmeans(noOfClusters)
    newcentres,cluster = cAlgo.fit(X)
    
    import matplotlib.pyplot as plt
    fig = plt.figure(7)
    ax = fig.add_subplot(1,1,1)
    markers=['*', 's', '<', 'o', 'X', '8', 'p', 'h', 'H', 'D', 'd', 'P']
    colorName = ['black','red','blue','green']
    for j in range(noOfClusters):
        index = np.flatnonzero(cluster == j)
        ax.scatter(X[index,0]  ,X[index,1],c=colorName[j],s=60,alpha=0.2,marker=markers[j])
    for i in range(len(cAlgo.fitHistory)):
        for j in range(noOfClusters):
            ax.text(cAlgo.fitHistory[i][j,0], cAlgo.fitHistory[i][j,1], str(i), style='italic',
                    bbox={'facecolor':'white', 'alpha':0.7, 'pad':2},color=colorName[j])
            #ax.annotate(str(i),xy=(cAlgo.fitHistory[i][j,0], cAlgo.fitHistory[i][j,1]),color=colorName[j])
            #ax.scatter(cAlgo.fitHistory[i][j,0], cAlgo.fitHistory[i][j,1],c='m',s=60,marker=markers[10+j])
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True,linestyle='-',color='0.75')
    ax.set_aspect('equal', 'datalim')
    