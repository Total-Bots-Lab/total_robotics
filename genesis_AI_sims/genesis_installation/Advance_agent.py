import genesis as gs
import numpy as np
import gym
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

# Joint names and indices
jnt_names = [
    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7',
    'finger_joint1', 'finger_joint2'
]
dofs_idx = [franka.get_joint(name).dof_start for name in jnt_names]

# Action space (joint limits)
action_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0], dtype=np.float32)
action_high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04], dtype=np.float32)
action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

# Observation space: joint positions + velocities + target position
obs_low = np.concatenate([action_low, -np.ones_like(action_low) * np.inf, action_low])
obs_high = np.concatenate([action_high, np.ones_like(action_high) * np.inf, action_high])
observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

def get_observation(target_position):
    positions = franka.get_dofs_position(dofs_idx)
    velocities = franka.get_dofs_velocity(dofs_idx)
    # If positions/velocities are torch tensors, move to CPU and convert to numpy
    if hasattr(positions, "cpu"):
        positions = positions.cpu().numpy()
    if hasattr(velocities, "cpu"):
        velocities = velocities.cpu().numpy()
    return np.concatenate([positions, velocities, target_position])

# Initial and target joint positions
initial_position = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04], dtype=np.float32)
target_position = np.array([2.0, 1.0, -2.0, -2.0, 2.0, 2.0, -2.0, 0.0, 0.04], dtype=np.float32)

# Reset robot to initial position
for _ in range(10):
    franka.set_dofs_position(initial_position, dofs_idx)
    scene.step()

# --- Advanced agent logic: proportional controller to target position ---
num_steps = 100
Kp = 1.0  # Proportional gain

current_position = initial_position.copy()

for step in range(num_steps):
    obs = get_observation(target_position)
    print(f"Step {step+1} observation:", obs)

    error = target_position - current_position
    action = current_position + Kp * error
    action = np.clip(action, action_low, action_high)

    franka.control_dofs_position(action, dofs_idx)

    for _ in range(50):  # More steps for visible movement
        scene.step()

    current_position = franka.get_dofs_position(dofs_idx)
    if hasattr(current_position, "cpu"):
        current_position = current_position.cpu().numpy()

# Cleanup
gs.destroy()
os._exit(0)