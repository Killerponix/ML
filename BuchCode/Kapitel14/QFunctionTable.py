import numpy as np

class qFunctionTable:
    def __init__(self,stateDim,actions,tablesize=[19,19], useRandom=False):
        t = (len(actions),) + tuple(tablesize)
        if useRandom: self._QFunction = np.random.rand( *t[:] )
        else: self._QFunction = np.zeros( t )
        
    def fit(self,state,action,Y):
        if len(state.shape) == 1 : state = state[ np.newaxis, :] #*\label{code:qtable:0}
        if np.isscalar(action) : action = np.array([action]) 
        row = np.rint(state[:,1]).astype(int).squeeze()
        col = np.rint(state[:,0]).astype(int).squeeze()
        a   = action.astype(int).squeeze()      
        self._QFunction[a,row,col] = Y.squeeze()

    def predict(self,state,action):
        if len(state.shape) == 1 : state = state[ np.newaxis, :] #*\label{code:qtable:1}
        if np.isscalar(action) : action = np.array([action])        
        row = np.rint(state[:,1]).astype(int).squeeze()
        col = np.rint(state[:,0]).astype(int).squeeze()
        a   = action.astype(int).squeeze()
        temp  = self._QFunction[a,row,col]
        if not np.isscalar(temp): temp  = temp.reshape(temp.shape[0],1)
        else: temp = np.array([temp])
        return temp 
