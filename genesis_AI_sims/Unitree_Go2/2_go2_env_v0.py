# -*- coding: utf-8 -*-
"""
Created on Sat May 17 15:42:21 2025

@author: ritwi
"""

import gymnasium as gym
#import genesis as gs
from libraries.genesis_setup import Genesis_Simulator 

'The main Env'
class Go2_Genesis_Env(gym.Env):
    def __init__(self):
        super().__init__()        
        
        'Genesis setup'
        Genesis_Simulator(self)
    
        
        'Define the Observation Space.'
        'Defining an arbitrary  Action Space for now.'
        self.action_space = gym.spaces.Discrete(1)
        
        'Define the Action Space.'
        'Defining an arbitrary Observation Space for now.'
        self.observation_space = gym.spaces.Discrete(1)


    def reset(self, *, seed=None, options=None):       # Need to study the seed part in details
        
        super().reset(seed=seed)                       # Need to study the seed part in details
        
        print('\nResetting the Env...')
        'Reload the scene to reset everything (Optional).'
        self.scene.reset()
        
        
        observation = self.observation_space.sample()
        info = {}
        
        return observation, info


    def step(self, action):
        # Step simulation
        self.scene.step()

        # Calculate observation, reward, done, info
        obs = self.observation_space.sample()
        reward = 0.0  # placeholder
        done = False  # placeholder
        info = {}

        return obs, reward, done, info


    def render(self, mode='human'):
        'Genesis has built-in viewer when show_viewer=True'
        pass  


    def close(self):
        pass
