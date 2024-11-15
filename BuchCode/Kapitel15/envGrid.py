import numpy as np
from environment  import environment
from learningAgentAdv import learningAgent

VERBOSE = False; epochen = 0 
np.random.seed(42)
#             012345678
gridworld = ['WWWWWWWWW', # 0
             'WgW    -W', # 1
             'W1W  W8-W', # 2
             'W1   W7gW', # 3
             'W5 WWW6-W', # 4
             'W  1   gW', # 5
             'WWWWWWWWW'] # 6

env = environment(gridworld,row=3, col=4)
observation = env.reset()
marvin = learningAgent(len(observation), actions=[0,1,2,3], vareps = 0.1, 
                       gamma=0.9, tau=0.05, alpha=0.1)     
count = 0
while epochen<5000:
    marvin.resetMemory()
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
        if VERBOSE: env.render()
    diff = marvin.learn()
    epochen += 1
import matplotlib.pyplot as plt
plt.plot(marvin.QHistory)


from visualFlex import plotQFunction 
plotQFunction(marvin,env,epochen)