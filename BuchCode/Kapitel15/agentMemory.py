import numpy as np
import pickle
from collections import deque

class agentMemory:
    def __init__(self, stateDim, memoryBuffer = 100000):
        self.buffer = deque(maxlen=memoryBuffer) 
        self.state     = np.NaN
        self.action    = np.NaN
        self.reward    = np.NaN
        self._nextstate = np.NaN
        
    def addState(self,state):
        self._nextstate = state
        self._addMemory()
        self.state = state       
        self._nextstate = np.NaN   

    def _addMemory(self):
        if np.any(np.isnan(self.state)) or np.any(np.isnan(self._nextstate)) or \
                  np.isnan(self.action) or np.isnan(self.reward):
            return False
        else:  
            newMemory = (self.state, self.action, self.reward, self._nextstate)
            self.buffer.append( newMemory )            
            return True

    def cleanCurrentMemory(self):
        self.state      = np.NaN
        self._nextstate = np.NaN
        self.action     = np.NaN
        self.reward     = np.NaN
                
    def getBatch(self, size=None):
        if size == None or size > len(self.buffer): size = len(self.buffer) 
        idx = np.random.choice(np.arange(len(self.buffer)), size=size, replace=False)
        b = [self.buffer[ii] for ii in idx]
        states = np.array([sample[0] for sample in b])
        actions = np.array([sample[1] for sample in b])
        rewards = np.array([sample[2] for sample in b])
        nextStates = np.array([sample[3] for sample in b])
        actions = actions[:,np.newaxis]
        rewards = rewards[:,np.newaxis]
        return (states,actions,rewards, nextStates)
    
    def saveMemory(self,name):
        filename = name+".mem"
        dbfile = open(filename, 'wb') 
        pickle.dump(self.buffer, dbfile)
        dbfile.close()

    def loadMemory(self,name):
        filename = name+".mem"
        dbfile = open(filename, 'rb') 
        self.buffer = pickle.load(dbfile)
        dbfile.close()
            