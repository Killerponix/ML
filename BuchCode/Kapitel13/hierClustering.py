import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

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
D = pdist(X, metric='euclidean')
Z = linkage(D, 'single', metric='euclidean')

plt.figure()
dendrogram(Z,truncate_mode='lastp',p=12, link_color_func=lambda k: 'k')
plt.ylabel('Distanz')
plt.gray()

plt.figure()
clusterNo = fcluster(Z, 0.5, criterion='distance')
(Number , counts) = np.unique(clusterNo,return_counts=True)
big4 = np.argsort(counts)[-4:]

denseCluster = np.flatnonzero(clusterNo==(big4[0]+1))
for i in range(1,len(big4)):
    index = np.flatnonzero(clusterNo==(big4[i]+1))
    denseCluster = np.hstack( (denseCluster,index) )    

plt.scatter(X[denseCluster,0],X[denseCluster,1], color='k')

outlierIndex = np.arange(X.shape[0])
outlierIndex = np.delete(outlierIndex, denseCluster)
plt.scatter(X[outlierIndex,0],X[outlierIndex,1], marker='+', c='r')

