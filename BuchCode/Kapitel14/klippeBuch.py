import numpy as np
from environment  import environment
from learningAgent import learningAgent
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
observation = env.reset()
marvin = learningAgent(len(observation), actions=[0,1,2,3], vareps = 0.5, gamma=0.99)    

epochen = 0; success = 0 
while True:
    observation = env.reset()   
    env.render()
    marvin.setSensor(observation)
    marvin.totalReward = 0
    done = False 
    while not done:
        action = marvin.getAction()
        observation, reward, done, steps = env.step(action)
        marvin.setReward(reward)          #*\label{code:klippe:0}
        marvin.setSensor(observation)     #*\label{code:klippe:1}
        marvin.learn()                    #*\label{code:klippe:2}
        env.render()
    epochen += 1
    if steps<8 and marvin.totalReward>0.93: #*\label{code:klippe:3} 
        marvin.vareps = max(marvin.vareps*0.9, 0.01)
        success += 1
        if success > 9: break
    else:
        success = 0 
print(epochen,marvin.vareps)