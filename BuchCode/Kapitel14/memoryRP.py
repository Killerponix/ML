import numpy as np

class memoryRP:
    def __init__(self, stateDim, memoryBuffer = 500000):
        self._stateMemory  = np.ones( (memoryBuffer,stateDim) )*np.NaN
        self._nextstateMemory  = np.ones( (memoryBuffer,stateDim) )*np.NaN
        self._actionMemory = np.ones(memoryBuffer)*np.NaN
        self._rewardMemory = np.ones(memoryBuffer)*np.NaN
        self._state     = np.NaN
        self._nextstate = np.NaN
        self._action    = np.NaN
        self._reward    = np.NaN
        self.memoryPos = 0 
        self._maxMemory = memoryBuffer
        self._buffersize = memoryBuffer
        self._stateDim = stateDim      
        
    def getState(self,idx): return self._stateMemory[idx,:]
    
    def getNextState(self,idx): return self._nextstateMemory[idx,:]

    def getAction(self,idx): return self._actionMemory[idx]
    
    def getReward(self,idx): return self._rewardMemory[idx]
        
    def addAction(self,action): self._action = action 
    
    def addReward(self,reward): self._reward = reward

    def addState(self,state):
        self._nextstate = state
        self._addMemory()
        self._state = state 
        self._nextstate = np.NaN # <----
        
    def cleanCurrentMemory(self):
        self._state     = np.NaN
        self._nextstate = np.NaN
        self._action    = np.NaN
        self._reward    = np.NaN
                
    def _addMemory(self):
        if np.any(np.isnan(self._state)) or np.any(np.isnan(self._nextstate)) or \
                  np.isnan(self._action) or np.isnan(self._reward):
            return(False)
        else:  
            idx1 = np.equal(self._stateMemory,self._state).all(1)
            idx2 = (self._actionMemory == self._action)
            out1 = np.logical_and(idx1,idx2)
            idx3 = np.equal(self._nextstateMemory,self._nextstate).all(1)
            out2 = np.logical_and(out1,idx3)
            
            if np.any(out2): # we have visited this point already 
                self.cleanCurrentMemory()
                return(False)
            
            self._stateMemory[self.memoryPos,:]     = self._state
            self._nextstateMemory[self.memoryPos,:] = self._nextstate
            self._actionMemory[self.memoryPos]      = self._action  
            self._rewardMemory[self.memoryPos]      = self._reward    
            
            self.memoryPos += 1
            return(True)

