import numpy as np
import pickle    
from QFunctionTable import qFunctionTable
from QFunctionANN import qFunctionANN #*\label{code:agentANN:0}
from agentMemory import agentMemory
from tensorflow.keras.models import load_model  #*\label{code:agentANN:1}

class learningAgent:
    def __init__(self,stateDim, actions, gamma=0.8, vareps = 0.01, tau=0.1, alpha=0.5,
                 learnbatch = None, memoryBuffer = 10000, tablesize=[19,19], 
                 ANN=False, networkArc = [100]):
        self.alpha  = alpha
        self.tau    = tau
        self._gamma = gamma
        self.vareps = vareps        
        self.totalReward = 0
        self._actionRange   = actions
        self.learnbatch = learnbatch
        self._M = agentMemory(stateDim, memoryBuffer = memoryBuffer)  
        self.ANN = ANN
        if self.ANN: self.QFunction = qFunctionANN(stateDim, actions,networkArc=networkArc)  #*\label{code:agentANN:2}
        else: self.QFunction = qFunctionTable(stateDim, actions,tablesize=tablesize)  #*\label{code:agentANN:3}
          
    def setSensor(self, state): self._M.addState(state)

    def resetMemory(self): self._M.cleanCurrentMemory()
     
    def setReward(self,reward): 
        self.totalReward += reward
        self._M.reward = reward 

    def getAction(self,observation=np.NaN):
        a      = self._chooseAction(observation)
        action = self._actionRange[a] 
        self._M.action = action
        return(action)

    def _chooseAction(self,observation=np.NaN):
        if np.any(np.isnan(observation)): observation = self._M.state
        if np.random.rand()<self.vareps: 
            choosenA = np.random.randint(0,len(self._actionRange))
            return(choosenA)
        qvalues = np.zeros(len(self._actionRange))
        for i in range(len(self._actionRange)):
            qvalues[i] = self.QFunction.predict(observation,self._actionRange[i])              
        toChoose = np.arange(0,len(qvalues))  
        qvalues = qvalues/self.tau - np.max(qvalues/self.tau)
        pW = np.exp(qvalues) / np.sum(np.exp(qvalues))
        if np.any(np.isnan(pW)) or np.any(np.isinf(pW)): 
            choosenA = np.random.randint(0,len(qvalues))
        else:
            choosenA = np.random.choice(toChoose,replace=False, p=pW)  
        return(choosenA)

    def learn(self):
        (state,action,reward, nextState) = self._M.getBatch(size=self.learnbatch)       #*\label{code:agent:exreplay:0}
        Qsa = self.QFunction.predict(state, action).squeeze()                           #*\label{code:agent:exreplay:1}
        maxQvalue = np.array([], dtype=np.float).reshape(reward.shape[0],0)             #*\label{code:agent:exreplay:2}
        for a in self._actionRange:                                                     #*\label{code:agent:exreplay:3}
            Qvalue = self.QFunction.predict(nextState, a*np.ones( (reward.shape[0],1))) #*\label{code:agent:exreplay:4}
            maxQvalue = np.hstack( (Qvalue,maxQvalue))                                  #*\label{code:agent:exreplay:5}
        maxQ = np.max(maxQvalue,axis=1).squeeze()                                       #*\label{code:agent:exreplay:6}
        Y = (1-self.alpha)*Qsa + self.alpha*(reward.squeeze() + self._gamma*maxQ )      #*\label{code:agent:exreplay:7}
        if np.max(np.abs(Y)) > 100:                                                     #*\label{code:agent:exreplay:8}
            print('Warning: Q-Fct seems to diverge! Max Value=',np.max(np.abs(Y)),flush=True) #*\label{code:agent:exreplay:9}
        self.QFunction.fit(state,action,Y) #*\label{code:agent:exreplay:10}
    
    def restore(self,name):
        self.loadMemory(name)
        self.loadQfunction(name)
        
    def save(self,name):
        self.saveMemory(name)
        self.saveQfunction(name)

    def saveMemory(self,name): 
        self._M.saveMemory(name)
    
    def loadMemory(self,name): 
        self._M.loadMemory(name)
    
    def saveQfunction(self,name):
        if self.ANN: 
            self.QFunction._QFunction.save(name, save_format='tf')
        else: 
            filename = name+".qfT"
            dbfile = open(filename, 'wb') 
            pickle.dump(self.QFunction, dbfile)
            dbfile.close()

    def loadQfunction(self,name):
        if self.ANN: 
            self.QFunction._QFunction = load_model(name)
            self.QFunction.unfitted = False
        else: 
            filename = name+".qfT"
            dbfile = open(filename, 'rb') 
            self.QFunction = pickle.load(dbfile)
            dbfile.close()
