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

# Joint setup
jnt_names = [
    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7',
    'finger_joint1', 'finger_joint2'
]
dofs_idx = [franka.get_joint(name).dof_start for name in jnt_names]

# Action space bounds
action_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0], dtype=np.float32)
action_high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04], dtype=np.float32)
action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

def get_observation():
    def to_numpy(x):
        return np.array(x.cpu().numpy(), dtype=np.float32) if hasattr(x, "cpu") else np.array(x, dtype=np.float32)

    positions = to_numpy(franka.get_dofs_position(dofs_idx))
    velocities = to_numpy(franka.get_dofs_velocity(dofs_idx))
    try:
        ee_pose = to_numpy(franka.get_link_pose("hand"))
    except AttributeError:
        ee_pose = np.zeros(7, dtype=np.float32)
    try:
        torques = to_numpy(franka.get_dofs_torque(dofs_idx))
    except AttributeError:
        torques = np.zeros_like(positions)
    return np.concatenate([positions, velocities, ee_pose, torques])

# Initial joint position
initial_position = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04], dtype=np.float32)

# --- Precompute reference trajectory ---
num_steps = 200
ref_traj = []
ref_joint_traj = []

for step in range(num_steps):
    target_position = initial_position.copy()
    target_position[0] = 1.5 * np.sin(0.05 * step)
    target_position[1] = 1.0 * np.cos(0.05 * step)
    target_position[7] = 0.02 + 0.02 * np.sin(0.1 * step)
    target_position[8] = 0.02 + 0.02 * np.cos(0.1 * step)

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

# Draw reference trajectory
scene.draw_debug_points(poss=ref_traj, colors=(1.0, 0.0, 0.0, 0.5))  # Red

# --- PD Controller Setup ---
Kp = 0.5
Kd = 0.05
current_position = initial_position.copy()
joint_traj = [initial_position.copy()]
draw_interval = 5
error_log = []

for step in range(num_steps):
    target_position = initial_position.copy()
    target_position[0] = 1.5 * np.sin(0.05 * step)
    target_position[1] = 1.0 * np.cos(0.05 * step)
    target_position[7] = 0.02 + 0.02 * np.sin(0.1 * step)
    target_position[8] = 0.02 + 0.02 * np.cos(0.1 * step)

    obs = get_observation()
    velocity = obs[len(current_position):len(current_position)*2]

    error = target_position - current_position
    action = current_position + Kp * error - Kd * velocity
    action = np.clip(action, action_low, action_high)

    franka.control_dofs_position(action, dofs_idx)

    # Multi-step simulation for better convergence
    for _ in range(3):
        scene.step()

    current_position = action
    joint_traj.append(current_position.copy())
    error_log.append(np.linalg.norm(error))

    # Visualization
    if step > 0 and (step % draw_interval == 0 or step == num_steps - 1):
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