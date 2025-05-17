# -*- coding: utf-8 -*-
"""
Created on Sat May 17 21:14:17 2025

@author: ritwi
"""

import gymnasium as gym
import time


'Registering the Gym Environment'
gym.register(id='Go2_Sim_Genesis', entry_point='main_gym_env_go2:Go2_Sim_Genesis')


test_agent_id = 'random_agent'
episodes = 10


'Calling the Environment'
env = gym.make('Go2_Sim_Genesis')    



'Play an Eplisode with a Random Agent'
for episode in range(episodes):
    
    'Starting countdown to measure the time taken for one episode'
    start = time.time()
    done = False
    obs = env.reset()
    print('\nObservation after Reset:', obs)
    total_reward = 0
    score = 0
    while not done:
        'Taking an random action'
        random_action = env.action_space.sample()
        
        'Getting the observation, reward and done status from the environment'
        obs, reward, done, info = env.step(random_action)

    
    end = time.time()
    print(f"Time taken for the episode: {(end-start)*10**3:.03f}ms")
    
