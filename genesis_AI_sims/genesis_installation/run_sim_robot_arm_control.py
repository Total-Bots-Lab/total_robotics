# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:50:03 2025

@author: sayantan
"""

# Import Genesis and OS
import genesis as gs
import os
import numpy as np  # For generating random actions
import gym

# Initializes Genesis with the GPU backend.
gs.init(backend=gs.gpu)

# Creates a new simulation scene
scene = gs.Scene(show_viewer=True)
print(dir(scene))
# Adds a flat ground plane to the scene
scene.add_entity(gs.morphs.Plane())

# Loads the Franka Emika Panda robot arm using its MJCF XML file
robot = gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
franka=scene.add_entity(robot)

# Finalizes the scene setup
scene.build()



jnt_names = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
    'finger_joint1',
    'finger_joint2',
]


dofs_idx = [franka.get_joint(name).dof_start for name in jnt_names]

# Get the list of controllable joint names
#joint_names = robot.joint_names
num_joints = len(jnt_names)


kp= 20.0
kd= 0.5

# Create an array of kp values for each motor degree of freedom (DOF)
kp_array = np.full(len(dofs_idx), kp)
# Print the kp array to verify its values
print('Proportional Gains Array:\n',kp_array)

# Create an array of kd values for each motor degree of freedom (DOF)
kd_array = np.full(len(dofs_idx), kd)
# Print the kd array to verify its values
print('Derivative Gains Array:\n',kd_array)

print('\nSetting Proportional and Derivative gains for each DOF...')
# Set the proportional gains (Kp) for the specified motor joints (motors_dof_idx)
franka.set_dofs_kp(kp_array, dofs_idx)

# Set the derivative gains (Kv/Kd) for the specified motor joints (motors_dof_idx)
franka.set_dofs_kv(kd_array, dofs_idx)

print('\nThe Proportional and Derivative gains of the DOFs are set')

print('\nRechecking the Proportional and Derivative gains for each DOF...')
print('\nThe Proportional gains of the DOFs are as follows')
print(franka.get_dofs_kp(dofs_idx))
print('\nThe Proportional gains of the DOFs are as follows')
print(franka.get_dofs_kv(dofs_idx))


# Joint order: joint1, joint2, joint3, joint4, joint5, joint6, joint7, finger_joint1, finger_joint2
action_low = np.array([
    -2.8973,   # joint1
    -1.7628,   # joint2
    -2.8973,   # joint3 (default)
    -3.0718,   # joint4
    -2.8973,   # joint5 (default)
    -0.0175,   # joint6
    -2.8973,   # joint7 (default)
    0.0,       # finger_joint1
    0.0        # finger_joint2
], dtype=np.float32)

action_high = np.array([
    2.8973,    # joint1
    1.7628,    # joint2
    2.8973,    # joint3 (default)
    -0.0698,   # joint4
    2.8973,    # joint5 (default)
    3.7525,    # joint6
    2.8973,    # joint7 (default)
    0.04,      # finger_joint1
    0.04       # finger_joint2
], dtype=np.float32)


action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

action = action_space.sample()
print("Sampled action:\n", action)


# Initial joint positions for Franka Emika Panda (from panda.xml "home" keyframe)
initial_position = np.array([
    0.0,        # joint1
    0.0,        # joint2
    0.0,        # joint3
   -1.57079,    # joint4
    0.0,        # joint5
    1.57079,    # joint6
   -0.7853,     # joint7
    0.04,       # finger_joint1
    0.04        # finger_joint2
], dtype=np.float32)


# Hard reset the initial position
for i in range(10):
    franka.set_dofs_position(initial_position, dofs_idx)
    scene.step()


# Try random actions using the control_dofs_position command
for i in range(100):
    action = action_space.sample()
    clipped_action = np.clip((initial_position+ action), action_low, action_high)
    print(clipped_action)
    franka.control_dofs_position(clipped_action, dofs_idx)
    
    for i in range(10):
        scene.step()

# # Setting up 5 simulation episodes
# for episode in range(5):
#     print('Starting Episode: ', episode)     # Print Episode Number
    
#     scene.reset()
    
#     # Run 100 time steps in each episode
#     for _ in range(100):
#         # Generate random joint positions within typical limits (-2.0 to 2.0 radians)
#         action = np.random.uniform(-2.0, 2.0, size=num_joints)

#         franka.control_dofs_position(action, dofs_idx)
#         # Send the action to the robot
#         #robot.set_joint_positions(action)
#         # Advances the physics engine by one step.
#         scene.step()
    
# Cleans up Genesis resources after the simulation is complete
gs.destroy()

# Reset kernel
os._exit(0)