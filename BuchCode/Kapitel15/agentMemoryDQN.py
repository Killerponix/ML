import numpy as np
import matplotlib.pyplot as plt

class agentMemory(object):
    def __init__(self, memSize, input_shape):
        self.memSize = memSize
        self.mCounter= 0
        self._stateM      = np.zeros((self.memSize, *input_shape))
        self._nextstateM  = np.zeros((self.memSize, *input_shape))
        self._actionM     = np.zeros(self.memSize, dtype=np.int32)
        self._rewardM     = np.zeros(self.memSize)
        self._doneM       = np.zeros(self.memSize, dtype=np.uint8)

    def addMemory(self, state, action, reward, nextState, done):
        idx = self.mCounter% self.memSize
        self.mCounter+= 1
        self._stateM[idx]      = state
        self._nextstateM [idx] = nextState
        self._actionM [idx]    = action
        self._rewardM[idx]     = reward
        self._doneM[idx]       = done
        
    def getBatch(self, bSize):
        maxMem     = min(self.mCounter, self.memSize)
        batchIdx   = np.random.choice(maxMem, bSize, replace=False)
        states     = self._stateM[batchIdx]
        actions    = self._actionM [batchIdx]
        rewards    = self._rewardM[batchIdx]
        nextStates = self._nextstateM [batchIdx]
        done       = self._doneM[batchIdx]
        return states, actions, rewards, nextStates, done
    
    def showMemory(self,no):
        print('Memory No.',no,' with memory counter',self.mCounter )
        print('Reward:',self._rewardM[no])
        print('Action', self._actionM[no])
        print('Done', self._doneM[no])
        fig = plt.figure()
        for i in range(4):
            ax = fig.add_subplot(1,4,i+1)
            ax.imshow(self._stateM[no,:,:,i])
        fig = plt.figure()
        for i in range(4):
            ax = fig.add_subplot(1,4,i+1)
            ax.imshow(self._nextstateM[no,:,:,i])
