# -*- coding: utf-8 -*-
"""
Created on Sat May 17 01:17:08 2025

@author: ritwik
"""

'Import Genesis and OS'
import genesis as gs
import os

'Initializes Genesis with the CPU backend.'
gs.init(backend=gs.cpu)

'Creates a new simulation scene'
scene = gs.Scene(show_viewer=True)

'Adds a flat ground plane to the scene'
scene.add_entity(gs.morphs.Plane())

'Loads the Franka Emika Panda robot arm using its MJCF XML file'
scene.add_entity(gs.morphs.MJCF(file='xml/unitree_go2/go2.xml'))

'Finalizes the scene setup'
scene.build()

'Setting up 5 simulation episodes'
for episode in range(5):
    print('Episode: ', episode)     # Print Episode Number
    
    'Run 100 time steps in each episode'
    for _ in range(100):
        
        'Advances the physics engine by one step.'
        scene.step()
    
'Cleans up Genesis resources after the simulation is complete'
gs.destroy()

'Reset karnel'
os._exit(00)