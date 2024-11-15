import numpy as np
import pickle    
from QFunctionTable import qFunctionTable
from QFunctionANN import qFunctionANN 
from agentMemory import agentMemory
from tensorflow.keras.models import load_model  

class learningAgentDoubleQ:
    def __init__(self,stateDim, actions, gamma=0.8, vareps = 0.01, tau=0.1, alpha=0.5,
                 learnbatch = 10000, memoryBuffer = 20000, 
                 networkArc1 = [100], networkArc2 = [100]): 
        self.alpha  = alpha
        self.tau    = tau
        self._gamma = gamma
        self.vareps = vareps        
        self.totalReward = 0
        self._actionRange   = actions
        self.learnbatch = learnbatch
        self._M = agentMemory(stateDim, memoryBuffer = memoryBuffer)  
        self.QFA= qFunctionANN(stateDim, actions,networkArc=networkArc1)  
        self.QFB= qFunctionANN(stateDim, actions,networkArc=networkArc2) 

    def _chooseAction(self,observation=np.NaN):
        if np.any(np.isnan(observation)): observation = self._M.state
        if np.random.rand()<self.vareps: 
            choosenA = np.random.randint(0,len(self._actionRange))
            return(choosenA)
        qvalues1 = np.zeros(len(self._actionRange))  #*\label{code:agent:dql:0}
        qvalues2 = np.zeros(len(self._actionRange))
        for i in range(len(self._actionRange)):
            qvalues1[i] = self.QFA.predict(observation,self._actionRange[i])
            qvalues2[i] = self.QFA.predict(observation,self._actionRange[i])  
        qvalues = (qvalues1 + qvalues2)/2 #*\label{code:agent:dql:1}
        toChoose = np.arange(0,len(qvalues))
        qvalues = qvalues/self.tau - np.max(qvalues/self.tau)
        pW = np.exp(qvalues) / np.sum(np.exp(qvalues))
        if np.any(np.isnan(pW)) or np.any(np.isinf(pW)): 
            choosenA = np.random.randint(0,len(qvalues))
        else:
            choosenA = np.random.choice(toChoose,replace=False, p=pW)  
        return(choosenA)

    def learn(self):
        (state,action,reward, nextState) = self._M.getBatch(size=self.learnbatch)
        maxQvalue = np.array([], dtype=np.float).reshape(reward.shape[0],0)
        if np.random.rand()<0.5:
            Q2Train   = self.QFA
            Q2Messure = self.QFB
        else:
            Q2Train   = self.QFB
            Q2Messure = self.QFA
        
        QTrainOld = Q2Train.predict(state, action).squeeze()  #*\label{code:agent:dql:2}
        for a in self._actionRange:                           #*\label{code:agent:dql:3}                          
            Qvalue = Q2Train.predict(nextState, a*np.ones( (reward.shape[0],1))) #*\label{code:agent:dql:4}
            maxQvalue = np.hstack( (Qvalue,maxQvalue))  #*\label{code:agent:dql:5}
        aMax = np.argmax(maxQvalue,axis=1) #*\label{code:agent:dql:6}
        maxQ = Q2Messure.predict(nextState, aMax[:,np.newaxis]).squeeze() #*\label{code:agent:dql:7}                                 
        Y = (1-self.alpha)*QTrainOld + self.alpha*(reward.squeeze() + self._gamma*maxQ ) #*\label{code:agent:dql:8}     
        if np.max(np.abs(Y)) > 100:                                                     
            print('Warning: Q-Fct seems to diverge! Max Value=',np.max(np.abs(Y)),flush=True) 
        Q2Train.fit(state,action,Y, epochs=1000) #*\label{code:agent:dql:9}  
 
    def saveQfunction(self,name):
        self.QFA._QFunction.save('QFA'+name, save_format='tf')
        self.QFB._QFunction.save('QFB'+name, save_format='tf')
        
    def loadQfunction(self,name):
        self.QFA._QFunction = load_model('QFA'+name)
        self.QFB._QFunction = load_model('QFB'+name)
        self.QFA.unfitted = False
        self.QFB.unfitted = False

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