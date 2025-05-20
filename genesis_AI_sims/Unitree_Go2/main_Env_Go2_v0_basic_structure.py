# -*- coding: utf-8 -*-
"""
Created on Sat May 17 15:42:21 2025

@author: ritwi
"""

import gymnasium as gym
import genesis as gs

'The main Env'
class Go2_Genesis_Env(gym.Env):
    def __init__(self):
        super().__init__()
        
        'Genesis setup'
        'Initializes Genesis with the CPU backend.(if not gs._initialized:)'
        gs.init(backend=gs.cpu)
        'Creates a new simulation scene. Set the simulation time step. Supressing FPS INFOs.'
        self.scene = gs.Scene(show_viewer=True,
                              sim_options=gs.options.SimOptions(dt=0.02,
                                                                gravity=(0, 0, -10),
                                                                ),
                              show_FPS=False)
        'Adds a flat ground plane to the scene.'
        self.scene.add_entity(gs.morphs.Plane())

        'Integrate the Go2 Robot xml.'
        self.robot = gs.morphs.MJCF(file="xml/Unitree_Go2/go2.xml")
        'Add an entity to the scene.'
        self.scene.add_entity(self.robot)
        'Builds the scene.'
        self.scene.build()
        
        'Define the Observation Space.'
        self.action_space = gym.spaces.Discrete(1)
        
        'Define the Action Space.'
        self.observation_space = gym.spaces.Discrete(1)


    def reset(self):
        
        print('\nResetting the Env...')
        'Reload the scene to reset everything (Optional).'
        self.scene.reset()
        
        return 0


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
