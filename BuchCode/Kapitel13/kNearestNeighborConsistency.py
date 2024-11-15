import numpy as np 
from scipy.spatial import KDTree

class kNearestNeighborConsistency:
    def __init__(self, feature, k=3):
        self.kdTree = KDTree(feature) 
        self.feature = feature
        self.k = k
        
    def consistency(self,c):
        clusterSet = np.unique(c)
        clusterIndex = 0
        for cluster in clusterSet:
            if cluster == -1: continue 
            idx = np.flatnonzero(c==cluster)
            numberOfElements = len(idx)
            (dist, neighbours) = self.kdTree.query(self.feature[idx,:],self.k)
            (clusters, counts) = np.unique(c[neighbours], return_counts=True)
            clusterIndex += counts[clusters==cluster]/numberOfElements 
        clusterIndex = float(clusterIndex / (self.k*clusterSet.shape[0]))
        return clusterIndex