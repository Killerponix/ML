import numpy as np
from tqdm import tqdm
from dqnAgent import dqnAgent
from pongEnvWrapper import envPhi 

env = envPhi('PongNoFrameskip-v4')
marvin = dqnAgent(gamma=0.99, vareps=1.0, lr=0.0001,
                  frameDims=(80,80,4), actions=[0,1,2,3,4,5], memSize=25000,
                  epsMin=0.02, bSize=32, replace=1000, epsDec=1e-5)
maxReward= -21
rewards = []; epsHistory = []
steps = 0
verbose = False

progress = tqdm(range(500),desc='Training',unit=' episode')
for epoche in progress:
    done = False
    env.reset()
    observation = np.zeros( (80,80,4) )
    totalReward = 0
    while not done:
        steps += 1
        action = marvin.getAction(observation)
        obs, reward, done, info = env.step(action)
        totalReward += reward
        marvin.addMemory(observation, action, reward, obs, int(done))
        if verbose : env.render()    
        marvin.learn()
        observation = obs

    rewards.append(totalReward)
    epsHistory.append(marvin.vareps)
    movingAvr = np.mean(rewards[-20:])
    msg  =' Training r='+str(totalReward)
    msg +=' vareps='+ str(round(marvin.vareps,ndigits=2))
    msg += ' avg='+str(movingAvr)    
    progress.set_description(msg)
    if epoche % 10 == 0: marvin.saveCNNs()
    if movingAvr>19: break

marvin.vareps = 0
done = False
env.reset()
observation = np.zeros( (80,80,4) )
totalReward = 0
while not done:
    steps += 1
    action = marvin.getAction(observation)
    obs, reward, done, info = env.step(action)
    totalReward += reward
    marvin.addMemory(observation, action, reward, obs, int(done))
    env.render()    
    observation = obs

  