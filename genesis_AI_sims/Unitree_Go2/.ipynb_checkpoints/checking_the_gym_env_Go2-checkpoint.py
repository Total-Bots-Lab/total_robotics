# -*- coding: utf-8 -*-
"""
Created on Sat May 17 21:14:17 2025

@author: ritwi
"""

import genesis as gs
import gymnasium as gym
import time
import os


'Registering and Calling the Gym Environment'
gym.register(id='Go2_Genesis_Env', entry_point='main_Env_Go2:Go2_Genesis_Env')
env = gym.make('Go2_Genesis_Env') 

'Number of Episodes'
episodes = 10


'Play Eplisodes with a Random Agent'
for episode in range(episodes):
    
    'Starting countdown to measure the time taken for one episode'
    start = time.time()
    done = False
    obs = env.reset()
    print('\nObservation after Reset:', obs)
    total_reward = 0
    score = 0
    step_count = 0
    
    while not done:
        'Taking an random action'
        random_action = env.action_space.sample()
        #print('\nAction taken by the agent:\n',random_action)
        
        'Getting the observation, reward and done status from the environment'
        obs, reward, done, info = env.step(random_action)
        
        step_count += 1

        if step_count >= 100:
            done = True

    
    end = time.time()
    print(f"\nTime taken for the episode: {(end-start)*10**3:.03f}ms")

'Cleans up Genesis resources after the simulation is complete'    
gs.destroy()
'Reset karnel'
os._exit(00)
