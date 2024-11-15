import numpy as np
from agentMemoryDQN import agentMemory
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

def qFunctionCNN(lr, outputs, frameDims):
    QFunction = Sequential()       
    QFunction.add(Conv2D(32, (8,8), strides=4, activation='relu', input_shape=frameDims))
    QFunction.add(Conv2D(64, (4,4), strides=2, activation='relu'))
    QFunction.add(Conv2D(64, (3,3), strides=1, activation='relu'))
    QFunction.add(Flatten())
    QFunction.add( Dense(512, activation='relu') )
    QFunction.add(Dense(outputs, activation='linear')) 
    QFunction.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')
    return QFunction

class dqnAgent(object):
    def __init__(self, lr, gamma, actions, vareps, bSize, frameDims,
                 replace=1000, epsDec=0.0,  epsMin=0.01,
                 memSize=10000, name='marvin'):
        self.actions = actions
        self.gamma   = gamma
        self.vareps  = vareps
        self.epsDec  = epsDec
        self.epsMin  = epsMin
        self.bSize   = bSize
        self.memory  = agentMemory(memSize, frameDims)
        self.replace = replace
        self.Q       = qFunctionCNN(lr, len(actions), frameDims)
        self.hatQ    = qFunctionCNN(lr, len(actions), frameDims)
        self.name    = name 
        self.steps   = 0

    def addMemory(self, state, action, reward, nextState, done):
        self.memory.addMemory(state, action, reward, nextState, done)

    def getAction(self, observation):
        if np.random.random() < self.vareps:
            action = np.random.choice(self.actions)
        else:
            actions = self.Q.predict(observation[np.newaxis,:])
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.memory.mCounter > self.bSize:
            if self.steps % self.replace == 0:
                self.hatQ.set_weights(self.Q.get_weights())
            
            state, action, r, nextState, done = self.memory.getBatch(self.bSize)
            hatQ    = self.hatQ.predict(nextState)
            hatQMax = np.max(hatQ, axis=1)
            idx     = np.arange(self.bSize)
            y       = self.Q.predict(state)
            y[idx, action] = r +  self.gamma*(1 - done)*hatQMax
            self.Q.train_on_batch(state, y)

            self.vareps = self.vareps - self.epsDec 
            if self.vareps < self.epsMin:
                self.vareps = self.epsMin
            self.steps += 1

    def saveCNNs(self):
        fname = self.name+'.h5'
        self.Q.save('Q'+fname)
        self.hatQ.save('hatQ'+fname)

    def loadCNNs(self):
        fname = self.name+'.h5'
        self.Q = load_model('Q'+fname)
        self.q_nexdt = load_model('hatQ'+fname)