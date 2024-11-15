import numpy as np

class qFunctionTable:
    def __init__(self,stateDim,actions,tablesize=[19,19], useRandom=False):
        t = (len(actions),) + tuple(tablesize)
        if useRandom: self._QFunction = np.random.rand( *t[:] )
        else: self._QFunction = np.zeros( t )
        
    def fit(self,state,action,Y):
        if len(state.shape) == 1 : state = state[ np.newaxis, :] 
        if np.isscalar(action) : action = np.array([action]) 
        a   = action.astype(int).squeeze()
        self._QFunction[(a,) + tuple(state.T)] = Y.squeeze() 

    def predict(self,state,action, mapMode=True):
        if len(state.shape) == 1 : state = state[ np.newaxis, :] 
        if np.isscalar(action) : action = np.array([action])        
        a   = action.astype(int).squeeze()
        states = np.rint(state).astype(int)
        temp  = self._QFunction[(a,) + tuple(states.T)]
        if not np.isscalar(temp): temp  = temp.reshape(temp.shape[0],1)
        else: temp = np.array([temp])
        return temp 
