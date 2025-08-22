'''
Requirement : Display actual vs reference trajectory of the arm movement 

Red Dots: Reference Trajectory
The red dots represent the reference trajectory—the desired path that the robot’s end-effector (tip) should follow.
This trajectory is precomputed using a combination of sine and cosine functions for the first two joints, creating a smooth, circular or elliptical path in 3D space.
These points are drawn once and remain static throughout the simulation, serving as a visual guide for the robot to follow.

Green/Blue Path: Actual Trajectory (Live)
The green/blue path (depending on Genesis’s default coloring) is the actual trajectory traced by the robot’s end-effector as the simulation progresses.
This path is updated live during the simulation loop: after each control step, the script clears previous debug visuals and redraws the actual trajectory up to the current point.
The robot’s controller (a proportional controller in your script) tries to move the joints so the end-effector follows the reference trajectory as closely as possible.

Robot Movement
The robot arm moves in real time, attempting to track the reference trajectory.
The actual trajectory may deviate from the reference due to controller limitations, joint limits, or simulation dynamics.

Summary
Red dots: The planned/reference path for the end-effector.
Green/blue path: The path actually followed by the robot, updated live as the simulation runs.
Robot: Moves in real time, showing how well it can follow the reference.
This setup allows you to visually compare the robot’s tracking performance against the desired trajectory as the simulation unfolds!

Author: Sayantan 
'''

import torch
import genesis as gs
import numpy as np
import gymnasium as gym
import os

# Initialize Genesis
gs.init(backend=gs.gpu)

# Create scene and add ground
scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())

# Load and add Franka robot
robot = gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
franka = scene.add_entity(robot)
scene.build()

jnt_names = [
    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7',
    'finger_joint1', 'finger_joint2'
]
dofs_idx = [franka.get_joint(name).dof_start for name in jnt_names]

action_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0], dtype=np.float32)
action_high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04], dtype=np.float32)
action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

def get_observation():
    positions = franka.get_dofs_position(dofs_idx)
    velocities = franka.get_dofs_velocity(dofs_idx)
    def to_numpy(x):
        if hasattr(x, "cpu"):
            x = x.cpu().numpy()
        return np.array(x, dtype=np.float32)
    positions = to_numpy(positions)
    velocities = to_numpy(velocities)
    try:
        ee_pose = franka.get_link_pose("hand")
        ee_pose = to_numpy(ee_pose)
    except AttributeError:
        ee_pose = np.zeros(7, dtype=np.float32)
    try:
        torques = franka.get_dofs_torque(dofs_idx)
        torques = to_numpy(torques)
    except AttributeError:
        torques = np.zeros_like(positions)
    return np.concatenate([positions, velocities, ee_pose, torques])

initial_position = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04], dtype=np.float32)

# --- Precompute reference trajectory for visualization ---
num_steps = 200
ref_traj = []
ref_joint_traj = []
for step in range(num_steps):
    target_position = initial_position.copy()
    target_position[0] = 1.5 * np.sin(0.05 * step)
    target_position[1] = 1.0 * np.cos(0.05 * step)
    franka.set_dofs_position(target_position, dofs_idx)
    scene.step()
    try:
        ee_pose = franka.get_link_pose("hand")
        ee_pos = ee_pose[:3] if hasattr(ee_pose, "__getitem__") else [0, 0, 0]
    except Exception:
        ee_pos = [0, 0, 0]
    ref_traj.append(ee_pos)
    ref_joint_traj.append(target_position.copy())

# Reset robot to initial position
for _ in range(10):
    franka.set_dofs_position(initial_position, dofs_idx)
    scene.step()

# --- Draw reference trajectory as red points ---
scene.draw_debug_points(poss=ref_traj, colors=(1.0, 0.0, 0.0, 0.5)) # red, semi-transparent

# --- Closed-loop reference tracking with moving target and joint trajectory trace ---
Kp = 0.2
current_position = initial_position.copy()
joint_traj = []
joint_traj.append(initial_position.copy())  # Add initial position

# Draw initial actual trajectory (just the starting point)
scene.clear_debug_objects()
scene.draw_debug_points(poss=ref_traj, colors=(1.0, 0.0, 0.0, 0.5))  # Red reference
joint_traj_tensor = torch.tensor(np.array(joint_traj, dtype=np.float32))
scene.draw_debug_path(
    qposs=joint_traj_tensor,
    entity=franka,
    link_idx=-1,
    density=1.0,
    frame_scaling=1.0
)

for step in range(num_steps):
    target_position = initial_position.copy()
    target_position[0] = 1.5 * np.sin(0.05 * step)
    target_position[1] = 1.0 * np.cos(0.05 * step)

    obs = get_observation()
    print(f"Step {step+1} observation:", obs)

    error = target_position - current_position
    action = current_position + Kp * error
    action = np.clip(action, action_low, action_high)

    franka.control_dofs_position(action, dofs_idx)
    for _ in range(10):
        scene.step()

    current_position = action
    joint_traj.append(current_position.copy())

    # --- Live update: clear previous and draw actual trajectory so far ---
    scene.clear_debug_objects()
    scene.draw_debug_points(poss=ref_traj, colors=(1.0, 0.0, 0.0, 0.5))  # Red reference
    joint_traj_tensor = torch.tensor(np.array(joint_traj, dtype=np.float32))
    scene.draw_debug_path(
        qposs=joint_traj_tensor,
        entity=franka,
        link_idx=-1,
        density=1.0,
        frame_scaling=1.0
    )

input("Press Enter to exit and close Genesis...")

gs.destroy()
os._exit(0)