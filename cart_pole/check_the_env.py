# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 03:28:48 2025

@author: ritwi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 2025

@author: ritwi
"""
import gymnasium as gym
import time
import math
import matplotlib.pyplot as plt

# Configuration
env_id = 'CartPole-v1'
test_agent_id = 'random_cartpole_agent'
episodes = 10

# Create the CartPole environment
env = gym.make(env_id, render_mode=None)  # Use "human" for rendering if needed

print('\nObservation Space:\n', env.observation_space)
print('\nSample Observation:\n', env.observation_space.sample())
print('\nSample Action:\n', env.action_space.sample())

# Test loop
reward_list = []
for episode in range(episodes):
    start = time.time()
    
    obs, info = env.reset(seed=None)
    print(f"\n--- Episode {episode+1} ---")
    print('Initial Observation:', obs)
    
    total_reward = 0
    done = False
    
    while not done:
        action = env.action_space.sample()  # Random agent
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward

        print('Action:', action)
        print('Reward:', reward)
        print('Next Observation:', obs)
        print('Done:', done)

    reward_list.append(total_reward)
    end = time.time()
    print(f"Time taken for the episode: {(end - start) * 1e3:.2f} ms")

# Plotting the reward histogram
plt.hist(reward_list, bins=10, color='green', edgecolor='black')
plt.xlabel('Total Reward per Episode')
plt.ylabel('Frequency')
plt.title('CartPole Reward Histogram')

# Save the plot
file_path = 'test_logs/' + test_agent_id + '/Reward_Histogram.jpeg'
import os
os.makedirs(os.path.dirname(file_path), exist_ok=True)
plt.savefig(file_path, dpi=500)

plt.show()
