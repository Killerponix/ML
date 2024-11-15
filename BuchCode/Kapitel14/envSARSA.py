SARSA = True ; VERBOSE = False; epochen = 0 
import numpy as np
from environment  import environment
if SARSA: from learningAgentAdvSARSA import learningAgent
else: from learningAgentAdv import learningAgent
np.random.seed(42)
#             012345
gridworld = ['WWWWWW', # 0
             'W    W', # 1
             'W   -W', # 2
             'W   -W', # 3
             'W   -W', # 4
             'W   gW', # 5
             'WWWWWW'] # 6

env = environment(gridworld,row=1, col=4)
observation = env.reset()
marvin = learningAgent(len(observation), actions=[0,1,2,3], vareps = 0.2, 
                       gamma=0.99, tau=0.05, alpha=0.2)     
while epochen<10000:
    observation = env.reset()  
    if VERBOSE: env.render()
    marvin.setSensor(observation)
    marvin.totalReward = 0
    done = False 
    Qalt = marvin.QFunction._QFunction.copy()
    while not done:
        action = marvin.getAction()
        observation, reward, done, steps = env.step(action)
        marvin.setReward(reward)          
        marvin.setSensor(observation)     
        marvin.learn()                    
        if VERBOSE: env.render()
    epochen += 1
 
     
print(epochen,marvin.vareps)

from visualFlex import plotQFunction 
plotQFunction(marvin,env,epochen)