import numpy as np
from environment  import environment
from learningAgentAdv import learningAgent
np.random.seed(42)
#                 012345
gridworld = ['WWWWWW', # 0
             'W    W', # 1
             'W   -W', # 2
             'W   -W', # 3
             'W   -W', # 4
             'W   gW', # 5
             'WWWWWW'] # 6

env = environment(gridworld,row=1, col=4)
env.empty = -0.1
env.wall  = -0.5
observation = env.reset()
marvin = learningAgent(len(observation), actions=[0,1,2,3], vareps = 1.0, 
                       gamma=0.99, tau=0.1, alpha=0.5)    

epochen = 0; success = 0 
while True:
    r = np.random.randint(1,6)
    c = np.random.randint(1,5)
    if c == 4 and r<1 : r=1
    observation = env.reset(row=r,col=c)  
    env.render()
    marvin.setSensor(observation)
    marvin.totalReward = 0
    done = False 
    while not done:
        action = marvin.getAction()
        observation, reward, done, steps = env.step(action)
        marvin.setReward(reward)          
        marvin.setSensor(observation)     
        marvin.learn()                    
        env.render()
    epochen += 1
    if steps<8 and marvin.totalReward>=1-steps*0.1:
        marvin.vareps = max(marvin.vareps*0.9, 0.01)
        success += 1
        if success > 19: break
    else:
        success = 0 
print(epochen,marvin.vareps)

from visualFlex import plotQFunction 
plotQFunction(marvin,env,epochen)