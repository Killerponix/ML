import numpy as np

def rescaleStates(obs,buckets=1,discrete=False):
    upper_bounds = np.array([ 2,  2.0,  0.25,  2.0])
    lower_bounds = np.array([-2, -2.0, -0.25, -2.0])
    obsNew = (obs - lower_bounds) / (upper_bounds - lower_bounds)  #*\label{code:cartenv:0}
    obsNew[obsNew>1] = 1  #*\label{code:cartenv:1}
    obsNew[obsNew<0] = 0  #*\label{code:cartenv:2}
    if discrete: obsNew = np.round(buckets*obsNew,0).astype(int) #*\label{code:cartenv:3-1}
    return obsNew

Gamma      = 0.99 
Tau        = 0.05
aufloesung = 20  #*\label{code:cartenv:3}
storeIt    = (aufloesung+1,aufloesung+1,aufloesung+1,aufloesung+1) #*\label{code:cartenv:4}
alphaInit  = 0.5; alphaLow = 0.1; delta=0.001 #*\label{code:cartenv:5}
varepsInit = 0.2; varepsLow = 0.001 #*\label{code:cartenv:6}
useANN     = False #*\label{code:cartenv:7}
networkArc = [100] #*\label{code:cartenv:8}
runsToStop = 20 #*\label{code:cartenv:9}
if useANN: 
    maxEpoch = 500
    discrete = False
else: 
    maxEpoch = 6000 #*\label{code:cartenv:10}
    discrete = True #*\label{code:cartenv:11}

def rewardFct(observation, rewardEnv=0):
    reward  = 0.25
    reward -= np.abs(observation[2])/0.15 
    reward -= (np.abs(observation[0]/2.40))**8   
    reward  = 0.1*reward
    return reward

import gym
from learningAgentAdv import learningAgent

def oneEpoch(agent,env, verbose=False,discrete=False,buckets=40):
    agent.resetMemory()
    observation = env.reset()  
    agent.setSensor(rescaleStates(observation,aufloesung,discrete=discrete))
    done = False 
    steps = 0 
    while not done:
        steps += 1
        action = agent.getAction()
        observation, reward, done, info = env.step(action)
        reward = rewardFct(observation)
        agent.setReward(reward)          
        agent.setSensor(rescaleStates(observation,buckets,discrete=discrete))   
        if verbose: env.render()
    return steps , done

import random
random.seed(42)
np.random.seed(42)
env = gym.make('CartPole-v0')
observation = env.reset()
marvin = learningAgent(len(observation), actions=[0,1], vareps = varepsInit, 
                       gamma=Gamma, tau=Tau, alpha=alphaInit, tablesize=storeIt,
                       ANN=useANN,networkArc=networkArc)     

from tqdm import tqdm
for epochen in tqdm(range(500),desc='Init Population',unit=' episode'): #*\label{code:cartenv:12}
    marvin.totalReward = 0    
    steps, done = oneEpoch(marvin,env,discrete=discrete, buckets=aufloesung)
    
perfect = 0; perfectBest=0
stepsDone = []
for epoche in tqdm(range(maxEpoch),desc='Training',unit=' episode'):
    marvin.totalReward = 0
    steps, done = oneEpoch(marvin,env,discrete=discrete, buckets=aufloesung)
    stepsDone.append(steps)
    for _ in range(5): marvin.learn() #*\label{code:cartenv:13}
    if steps > 190: 
        perfect += 1
        if perfect > perfectBest:
            perfectBest = perfect
            lastSave = 'agent'+str(epoche)
            marvin.save(lastSave)
    else: 
        perfect = 0
    if perfect>runsToStop: break
    if steps > 100:
        marvin.alpha  = max(marvin.alpha -delta,alphaLow)
        marvin.vareps = max(marvin.vareps-delta,varepsLow)
    else: 
        marvin.alpha  = min(marvin.alpha +delta,alphaInit)
        marvin.vareps = min(marvin.vareps+delta,varepsInit)
env.close()

del marvin
marvin = learningAgent(len(observation), actions=[0,1], vareps = varepsInit, 
                        gamma=Gamma, tau=Tau, alpha=alphaInit, tablesize=storeIt,
                        ANN=useANN,networkArc=networkArc)   
marvin.restore(lastSave)

env = gym.make('CartPole-v0')
marvin.vareps = 0 
allSteps = 0
for i in range(100):
    steps, done = oneEpoch(marvin,env,discrete=discrete, buckets=aufloesung,verbose=True)
    allSteps += steps
    print(i,steps)
env.close()        
print(allSteps/100)

import matplotlib.pyplot as plt
plt.figure()
plt.title('Alpha')
plt.plot(mAlpha,c='k')
plt.figure()
plt.title('Total Reward')
plt.plot(totalReward,c='k')
plt.figure()
plt.title('Steps Done per Episode')
x = np.arange(len(stepsDone))
p = np.poly1d( np.polyfit(x,stepsDone,deg=1) )
plt.scatter(x,stepsDone,c='r')
plt.plot(x, p(x),c='k',lw=4)

