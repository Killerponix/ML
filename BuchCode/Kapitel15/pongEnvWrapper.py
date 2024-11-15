import numpy as np
import gym

class Skip4Env(gym.Wrapper):
    def step(self, action):
        reward = 0.0
        done = False
        for _ in range(4):
            obs, r, done, info = self.env.step(action)
            reward += r
            if done: break
        return obs, reward, done, info

class GrayCrop(gym.ObservationWrapper):
    def observation(self, frame):
        processedFrame = np.reshape(frame, frame.shape).astype(np.float32)
        processedFrame = np.dot(processedFrame[...,:3], [0.299, 0.587, 0.114])
        processedFrame = processedFrame[35:195:2, ::2].reshape(80,80,1)
        return processedFrame.astype(np.uint8)
        
class BufferWrapper(gym.ObservationWrapper):
    def reset(self):
        self.buffer = np.zeros( (80,80,4))
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:,:,0:3] = self.buffer[:,:,1:4]
        self.buffer[:,:,3] = observation.squeeze()
        obs = np.array(self.buffer).astype(np.float32) / 255.0
        return obs

def envPhi(envName):
    env = gym.make(envName)
    env = Skip4Env(env)
    env = GrayCrop(env)
    env = BufferWrapper(env)
    return env
 