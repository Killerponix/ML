import numpy as np
from QFunctionTable import qFunctionTable
    
class learningAgent:
    def __init__(self,stateDim, actions, gamma=0.8, vareps = 0.01, tau=0.1, alpha=0.5):
        self.alpha = alpha
        self.tau   = tau
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
        nextA  = self._chooseAction(observation=self._nextState)
        qValue = self.QFunction.predict(self._nextState, nextA) 
        Qalt = self.QFunction.predict(self._state,self._action)
        QNeu = (1.0-self.alpha)*Qalt + self.alpha*(self._reward + self._gamma*qValue) 
        self.QFunction.fit(self._state, self._action, QNeu) 

    def getAction(self,observation=np.NaN):
        a      = self._chooseAction(observation)
        action = self._actionRange[a] 
        self._action = action
        return(action)

    def _chooseAction(self,observation=np.NaN):
        if np.any(np.isnan(observation)): observation = self._nextState
        if np.random.rand()<self.vareps: 
            choosenA = np.random.randint(0,len(self._actionRange))
            return(choosenA)
        qvalues = np.zeros(len(self._actionRange))
        for i in range(len(self._actionRange)):
            qvalues[i] = self.QFunction.predict(observation,self._actionRange[i])              
        toChoose = np.arange(0,len(qvalues))  #*\label{code:laAdv:0}
        qvalues = qvalues/self.tau - np.max(qvalues/self.tau)
        pW = np.exp(qvalues) / np.sum(np.exp(qvalues))
        if np.any(np.isnan(pW)) or np.any(np.isinf(pW)): #*\label{code:laAdv:1}
            choosenA = np.random.randint(0,len(qvalues))
        else:
            choosenA = np.random.choice(toChoose,replace=False, p=pW)  #*\label{code:laAdv:2}
        return(choosenA)
