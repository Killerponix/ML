import numpy as np
from scipy.spatial import KDTree

class DBSCAN:
    def __init__(self, X, p=2, leafSize=30):
        self.p = p
        self.leafSize = leafSize
        self.X = X
        self.kdTree = KDTree(self.X, leafsize=self.leafSize) 

    def fit_predict(self,eps=0.5, MinPts=5):
        self.eps = eps
        self.MinPts = MinPts
        C = 0 
        self.visited = np.zeros(self.X.shape[0],dtype=bool)
        self.clusters = -10*np.ones(self.X.shape[0],dtype=int)
        for i in range(self.visited.shape[0]):
            if not self.visited[i]:
                self.visited[i] = True 
                P = self.X[i,:]
                N = np.array(self.kdTree.query_ball_point(P, self.eps, p=self.p))
                if N.shape[0] < self.MinPts:
                    self.clusters[i] = -1
                else:
                    C = C+ 1
                    self.visited[i] = C
                    self._expandCluster(N, C)
        return self.clusters
    
    def _expandCluster(self,N, C):
        elements = N.shape[0]
        j = 0
        while j < elements:
            i = N[j]
            if not self.visited[i]:
                self.visited[i] = True
                NExpend =np.array(self.kdTree.query_ball_point(self.X[i,:],self.eps,p=self.p))
                if NExpend.shape[0] >= self.MinPts:
                    N = np.hstack( (N,NExpend) )
                    elements = N.shape[0]
            if self.clusters[i]<0:
                self.clusters[i] = C 
            j = j + 1
            
    def analyseEps(self,MinPts=5):
        (d,_) = self.kdTree.query(self.X,k=MinPts, p=self.p)
        d = np.max(d,axis=1)
        return d
            
if __name__ == '__main__':
    
    def twoMoonsProblem( SamplesPerMoon=240, pNoise=2):
        np.random.seed(42) 
        tMoon0 = np.linspace(0, np.pi, SamplesPerMoon)
        tMoon1 = np.linspace(0, np.pi, SamplesPerMoon)
        Moon0x = np.cos(tMoon0)
        Moon0y = np.sin(tMoon0)
        Moon1x = 1 - np.cos(tMoon1)
        Moon1y = 0.5 - np.sin(tMoon1) 
        X = np.vstack((np.append(Moon0x, Moon1x), np.append(Moon0y, Moon1y))).T
        X = X + pNoise/100*np.random.normal(size=X.shape)
        Y = np.hstack([np.zeros(SamplesPerMoon), np.ones(SamplesPerMoon)])
        return X, Y

    (XMoons,_) = twoMoonsProblem()
    
    clusterAlg = DBSCAN(XMoons)
    
    import matplotlib.pyplot as plt
    d = clusterAlg.analyseEps()
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    ax.hist(d,10,normed=1, facecolor='k', alpha=0.5)
    ax.set_xlabel('Distanz')
     
    c = clusterAlg.fit_predict(eps=0.07,MinPts=5)
    
    index = np.flatnonzero(c == -1)
    print(index.shape[0]/c.shape[0])
    
    markers=['*', 's', '<', 'o', 'X', '^', 'h', 'H', 'D', 'd', 'P']
    colorName = ['red','black','blue','green', 'c', 'm', 'y']   
    fig = plt.figure(2)
    ax = fig.add_subplot(1,1,1)
    
    for i in range(0,c.max()):
        index = np.flatnonzero(c == i+1)
        ax.scatter(XMoons[index,0],XMoons[index,1],c=colorName[i],s=60,alpha=0.2,marker=markers[i])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True,linestyle='-',color='0.75')
    ax.set_aspect('equal', 'datalim')
    ax.set_title("Two Moons Set")
    
    
    
    
    