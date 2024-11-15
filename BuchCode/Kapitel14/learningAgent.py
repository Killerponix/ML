import numpy as np
from QFunctionTable import qFunctionTable
    
class learningAgent:
    def __init__(self,stateDim, actions, gamma=0.8, vareps = 0.01, tau=1, alpha=0.5):
        self._actionRange   = actions
        self._gamma = gamma
        self.vareps = vareps
        self.QFunction = qFunctionTable(stateDim, actions)
        self.totalReward = 0          
        self._nextState = None
        self._state     = None
        self._action    = None
        self._reward    = None

    def setSensor(self, state): 
        self._state = self._nextState
        self._nextState    = state
     
    def setReward(self,reward): 
        self.totalReward += reward
        self._reward = reward
    
    def learn(self):
        maxQvalue = np.zeros(len(self._actionRange))
        for i, a in enumerate(self._actionRange):
            maxQvalue[i] = self.QFunction.predict(self._nextState, a) #*\label{code:lagent1:0}
        maxQ = np.max(maxQvalue).squeeze()
        Y = self._reward + self._gamma*maxQ #*\label{code:lagent1:1}
        self.QFunction.fit(self._state, self._action, Y) #*\label{code:lagent1:2}

    def getAction(self,observation=np.NaN):
        a      = self._chooseAction(observation)
        action = self._actionRange[a] 
        self._action = action
        return(action)

    def _chooseAction(self,observation=np.NaN):
        if np.any(np.isnan(observation)): observation = self._nextState
        if np.random.rand()<self.vareps: #*\label{code:lagent1:3}
            choosenA = np.random.randint(0,len(self._actionRange))
            return(choosenA)
        qvalues = np.zeros(len(self._actionRange))
        for i in range(len(self._actionRange)):
            qvalues[i] = self.QFunction.predict(observation,self._actionRange[i])  #*\label{code:lagent1:4}            
        choosenA = np.argwhere(qvalues == np.max(qvalues))  #*\label{code:lagent1:5}
        if choosenA.shape[0] != 1:
            idx = np.random.randint(0,choosenA.shape[0]) #*\label{code:lagent1:6} 
            choosenA = choosenA.squeeze()[idx]
        else:
            choosenA = choosenA.squeeze()
        return(choosenA)
    