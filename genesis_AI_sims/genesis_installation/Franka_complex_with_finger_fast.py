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
    # --- Advanced finger joint movement: sinusoidal open/close ---
    target_position[7] = 0.02 + 0.02 * np.sin(0.1 * step)  # finger_joint1
    target_position[8] = 0.02 + 0.02 * np.cos(0.1 * step)  # finger_joint2
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
scene.draw_debug_points(poss=ref_traj, colors=(1.0, 0.0, 0.0, 0.5))  # Red reference
joint_traj_tensor = torch.tensor(np.array(joint_traj, dtype=np.float32))
scene.draw_debug_path(
    qposs=joint_traj_tensor,
    entity=franka,
    link_idx=-1,
    density=1.0,
    frame_scaling=1.0
)

draw_interval = 5  # Draw every 5 steps

for step in range(num_steps):
    target_position = initial_position.copy()
    target_position[0] = 1.5 * np.sin(0.05 * step)
    target_position[1] = 1.0 * np.cos(0.05 * step)
    # --- Advanced finger joint movement: sinusoidal open/close ---
    target_position[7] = 0.02 + 0.02 * np.sin(0.1 * step)  # finger_joint1
    target_position[8] = 0.02 + 0.02 * np.cos(0.1 * step)  # finger_joint2

    obs = get_observation()
    print(f"Step {step+1} observation:", obs)

    error = target_position - current_position
    action = current_position + Kp * error
    action = np.clip(action, action_low, action_high)

    franka.control_dofs_position(action, dofs_idx)
    scene.step()  # Only one step for speed

    current_position = action
    joint_traj.append(current_position.copy())

    # --- Live update: clear previous and draw actual trajectory so far ---
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