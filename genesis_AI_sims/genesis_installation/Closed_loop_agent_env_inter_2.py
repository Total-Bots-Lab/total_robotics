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

# Observation space: joint positions + velocities + ee_pose + torques
obs_low = np.concatenate([
    action_low,
    -np.ones_like(action_low) * np.inf,
    np.array([-np.inf]*7, dtype=np.float32),
    -np.ones_like(action_low) * np.inf
])
obs_high = np.concatenate([
    action_high,
    np.ones_like(action_high) * np.inf,
    np.array([np.inf]*7, dtype=np.float32),
    np.ones_like(action_high) * np.inf
])
observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

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

# Initial joint positions
initial_position = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04], dtype=np.float32)

# Reset robot to initial position
for _ in range(10):
    franka.set_dofs_position(initial_position, dofs_idx)
    scene.step()

# --- Closed-loop reference tracking with moving target ---
num_steps = 200
Kp = 0.2  # Lower gain for smoother motion

current_position = initial_position.copy()

for step in range(num_steps):
    # Example: oscillate joint1 and joint2 as a moving target
    target_position = initial_position.copy()
    target_position[0] = 1.5 * np.sin(0.05 * step)
    target_position[1] = 1.0 * np.cos(0.05 * step)

    obs = get_observation()
    print(f"Step {step+1} observation:", obs)

    # Proportional control towards target
    error = target_position - current_position
    action = current_position + Kp * error
    action = np.clip(action, action_low, action_high)

    franka.control_dofs_position(action, dofs_idx)

    for _ in range(10):
        scene.step()

    current_position = action

# Cleanup
gs.destroy()
os._exit(0)