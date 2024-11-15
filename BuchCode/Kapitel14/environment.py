import copy
import numpy as np
        
class environment:
    def __init__(self, gridworld,row,col):
        self.wall  = -0.1    # 'W'
        self.minus = -1      # '-'
        self.empty = -0.01   # ' '
        self.plus  = 0       # '+'
        self.goal  = 1       # 'g'
        self.world = gridworld        
        self.reset(row,col)            

    def reset(self, row=-1, col=-1):
        if row != -1:
            self.initCol = col; self.initRow = row #*\label{code:maze:0}
        self.row = self.initRow; self.col = self.initCol #*\label{code:maze:1}
        self.steps = 0             
        observation = np.array([self.col, self.row])
        return observation
        
    def step(self,action):
        done = False 
        self.steps += 1    
        info = self.steps
        row = self.row; col = self.col #*\label{code:maze:2}
    
        if self.world[row][col] == 'g': #*\label{code:maze:3} 
            done = True
            reward = 0 
            observation = np.array([self.col,self.row])
            return observation, reward, done, info #*\label{code:maze:4}
              
        if   action == 0: row = self.row+1 # south
        elif action == 1: col = self.col+1 # east 
        elif action == 2: row = self.row-1 # north    
        elif action == 3: col = self.col-1 # west
        newPlace = self.world[row][col]
        if ord(newPlace)>48 and ord(newPlace)<58:
            if np.random.rand()<float(newPlace)/10: newPlace = 'W'
            else: newPlace = ' '
        if newPlace == 'W':
            reward = self.wall 
            row = self.row
            col = self.col
        else: 
            self.row = row
            self.col = col
            if newPlace   == '-': reward = self.minus
            elif newPlace == '+': reward = self.plus
            elif newPlace == ' ': reward = self.empty                
            elif newPlace == 'g':
                reward = self.goal
                done = True
        
        observation = np.array([self.col,self.row])
        return observation, reward, done, info 
        
    def render(self):
        w = copy.deepcopy(self.world)
        w[self.row] = w[self.row][0:self.col] + 'O' + w[self.row][self.col+1:len(self.world)]
        for line in w: print(line)
        print()

if __name__ == '__main__':
    from visualFlex import plotQFunction 
    
    np.random.seed(42)
    from learningAgent import learningAgent
#                 012345
    gridworld = ['WWWWWW', # 0
                 'W    W', # 1
                 'W   -W', # 2
                 'W   -W', # 3
                 'W   -W', # 4
                 'W   gW', # 5
                 'WWWWWW'] # 6

    nurObenRechts = True
    maxSuccess = 9 if nurObenRechts else 200
    
    epochen = 0
    env = environment(gridworld,row=1, col=4)
    observation = env.reset()
    marvin = learningAgent(len(observation), actions=[0,1,2,3], vareps = 1.0, gamma=0.99)    
    plotQFunction(marvin,env, epochen)
    
    N = 0 
    while N<20:
        r = np.random.randint(1,6)
        c = np.random.randint(1,5)
        observation = env.reset(row=r,col=c)    
        marvin.setSensor(observation)
        done = False; totalReward = 0
        Qalt = marvin.QFunction._QFunction.copy()
        while not done:
            action = marvin.getAction()
            observation, reward, done, steps = env.step(action)
            marvin.setReward(reward)
            marvin.setSensor(observation)
            marvin.learn()
            totalReward += reward
        epochen += 1
        #diff = np.max(np.abs(Qalt  - marvin.QFunction._QFunction))
        diff = np.linalg.norm( (Qalt  - marvin.QFunction._QFunction).flatten(),ord=1)
        print(diff)
        if diff<0.0001: N += 1
        else: N = 0 
            
        if epochen%20 == 0: plotQFunction(marvin,env,epochen)
        
    plotQFunction(marvin,env,epochen)
    observation = env.reset(row=1, col=4)            
    marvin.setSensor(observation)
    env.render()
    done = False
    totalReward = 0
    marvin.vareps = 0
    while not done:
        action = marvin.getAction()
        observation, reward, done, info = env.step(action)
        marvin.setReward(reward)
        marvin.setSensor(observation)
        env.render()
        totalReward += reward
        print(info, totalReward, reward, action, done)
        if info > 100: done = True
        print()
