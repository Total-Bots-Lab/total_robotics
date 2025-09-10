"""
Robot Configuration Examples
===========================

This file shows how to configure different robots for the universal platform.
"""

import numpy as np
from typing import Dict, List, Optional, Any

from unified_platform.config.universal_config import RobotLibrary

# =============================================================================
# CONFIGURATION EXAMPLES FOR DIFFERENT ROBOTS
# =============================================================================

# Example 1: Franka Panda Arm (7-DOF manipulation)
franka_config = RobotLibrary.custom(
    name="franka_panda",
    urdf_path="urdf/franka_panda/panda.urdf",  # Your URDF path
    joint_names=[
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7"
    ],
    default_joint_angles={
        "panda_joint1": 0.0,
        "panda_joint2": -0.785,
        "panda_joint3": 0.0,
        "panda_joint4": -2.356,
        "panda_joint5": 0.0,
        "panda_joint6": 1.571,
        "panda_joint7": 0.785,
    },
    base_init_pos=[0.0, 0.0, 0.0],  # On ground
    kp=150.0,  # Stiffer for precise manipulation
    kd=10.0,
    action_scale=0.05,  # Small movements for precision
)

# Example 2: UR5 Robot Arm (6-DOF industrial arm)
ur5_config = RobotLibrary.custom(
    name="ur5",
    urdf_path="urdf/ur5/ur5.urdf",
    joint_names=[
        "shoulder_joint", "upper_arm_joint", "forearm_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ],
    default_joint_angles={
        "shoulder_joint": 0.0,
        "upper_arm_joint": -1.57,
        "forearm_joint": 1.57,
        "wrist_1_joint": -1.57,
        "wrist_2_joint": -1.57,
        "wrist_3_joint": 0.0,
    },
    base_init_pos=[0.0, 0.0, 0.1],
    kp=100.0,
    kd=5.0,
    action_scale=0.1,
)

# Example 3: ANYmal Quadruped (12-DOF locomotion)
anymal_config = RobotLibrary.custom(
    name="anymal",
    urdf_path="urdf/anymal/anymal.urdf",
    joint_names=[
        # Front left leg
        "LF_HAA", "LF_HFE", "LF_KFE",
        # Front right leg  
        "RF_HAA", "RF_HFE", "RF_KFE",
        # Hind left leg
        "LH_HAA", "LH_HFE", "LH_KFE",
        # Hind right leg
        "RH_HAA", "RH_HFE", "RH_KFE",
    ],
    default_joint_angles={
        # Front legs
        "LF_HAA": 0.0, "LF_HFE": 0.4, "LF_KFE": -0.8,
        "RF_HAA": 0.0, "RF_HFE": 0.4, "RF_KFE": -0.8,
        # Hind legs
        "LH_HAA": 0.0, "LH_HFE": -0.4, "LH_KFE": 0.8,
        "RH_HAA": 0.0, "RH_HFE": -0.4, "RH_KFE": 0.8,
    },
    base_init_pos=[0.0, 0.0, 0.5],  # Standing height
    kp=20.0,  # Softer for dynamic locomotion
    kd=0.5,
    action_scale=0.25,
)

# Example 4: Custom Simple Robot (4-DOF arm)
simple_arm_config = RobotLibrary.custom(
    name="simple_arm",
    urdf_path="urdf/simple_arm/arm.urdf",
    joint_names=["base_joint", "shoulder_joint", "elbow_joint", "wrist_joint"],
    default_joint_angles={
        "base_joint": 0.0,
        "shoulder_joint": 0.5,
        "elbow_joint": -1.0,
        "wrist_joint": 0.5,
    },
    base_init_pos=[0.0, 0.0, 0.2],
    kp=50.0,
    kd=2.0,
    action_scale=0.2,
)

# =============================================================================
# REWARD CONFIGURATION EXAMPLES
# =============================================================================

# Manipulation task rewards
manipulation_rewards = [
    {"type": "position_tracking", "weight": 10.0, "params": {"target_pos": [0.5, 0.0, 0.3]}},
    {"type": "action_smoothness", "weight": -0.1},
    {"type": "joint_regularization", "weight": -0.01},
    {"type": "energy_penalty", "weight": -0.001},
]

# Locomotion task rewards  
locomotion_rewards = [
    {"type": "forward_velocity", "weight": 5.0, "params": {"target_velocity": 1.0}},
    {"type": "upright_orientation", "weight": 2.0},
    {"type": "action_smoothness", "weight": -0.1},
    {"type": "joint_regularization", "weight": -0.01},
    {"type": "foot_contact", "weight": 1.0},
]

# General exploration rewards
exploration_rewards = [
    {"type": "action_smoothness", "weight": -0.1},
    {"type": "joint_regularization", "weight": -0.01},
    {"type": "survival", "weight": 1.0},
]

# =============================================================================
# OBSERVATION CONFIGURATION EXAMPLES  
# =============================================================================

# Full state observation
full_obs = [
    {"name": "joint_pos", "size": None, "scale": 1.0},      # Auto-detected
    {"name": "joint_vel", "size": None, "scale": 0.1},     # Auto-detected
    {"name": "base_pos", "size": 3, "scale": 1.0},
    {"name": "base_orn", "size": 4, "scale": 1.0},         # Quaternion
    {"name": "base_vel", "size": 3, "scale": 0.1},
    {"name": "base_ang_vel", "size": 3, "scale": 0.1},
    {"name": "actions", "size": None, "scale": 1.0},       # Previous actions
]

# Minimal observation
minimal_obs = [
    {"name": "joint_pos", "size": None, "scale": 1.0},
    {"name": "joint_vel", "size": None, "scale": 0.1},
]

# Manipulation-focused observation
manipulation_obs = [
    {"name": "joint_pos", "size": None, "scale": 1.0},
    {"name": "joint_vel", "size": None, "scale": 0.1},
    {"name": "end_effector_pos", "size": 3, "scale": 1.0},
    {"name": "end_effector_vel", "size": 3, "scale": 0.1},
    {"name": "actions", "size": None, "scale": 1.0},
]

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def create_franka_env():
    """Create Franka Panda environment for pick-and-place tasks."""
    from unified_platform.environment.generic_robot_env import GenericRobotGymEnv
    from unified_platform.config.reward_system import create_custom_rewards
    
    reward_manager = create_custom_rewards(manipulation_rewards)
    
    return GenericRobotGymEnv(
        robot_config=franka_config,
        reward_manager=reward_manager,
        obs_components=manipulation_obs,
        render_mode="human"
    )

def create_ur5_env():
    """Create UR5 environment for industrial tasks."""
    from unified_platform.environment.generic_robot_env import GenericRobotGymEnv
    from unified_platform.config.reward_system import create_custom_rewards
    
    reward_manager = create_custom_rewards(manipulation_rewards)
    
    return GenericRobotGymEnv(
        robot_config=ur5_config,
        reward_manager=reward_manager,
        obs_components=minimal_obs,
        render_mode=None
    )

def create_anymal_env():
    """Create ANYmal environment for locomotion tasks."""
    from unified_platform.environment.generic_robot_env import GenericRobotGymEnv
    from unified_platform.config.reward_system import create_custom_rewards
    
    reward_manager = create_custom_rewards(locomotion_rewards)
    
    return GenericRobotGymEnv(
        robot_config=anymal_config,
        reward_manager=reward_manager,
        obs_components=full_obs,
        render_mode="human"
    )

# =============================================================================
# QUICK START FUNCTIONS
# =============================================================================

def get_config_for_robot(robot_name: str):
    """Get predefined configuration for common robots."""
    configs = {
        "franka": franka_config,
        "ur5": ur5_config,
        "anymal": anymal_config,
        "simple_arm": simple_arm_config,
    }
    return configs.get(robot_name)

def get_rewards_for_task(task_type: str):
    """Get predefined rewards for common task types."""
    rewards = {
        "manipulation": manipulation_rewards,
        "locomotion": locomotion_rewards,
        "exploration": exploration_rewards,
    }
    return rewards.get(task_type, exploration_rewards)

def get_obs_for_task(task_type: str):
    """Get predefined observations for common task types."""
    observations = {
        "manipulation": manipulation_obs,
        "locomotion": full_obs,
        "minimal": minimal_obs,
        "full": full_obs,
    }
    return observations.get(task_type, minimal_obs)
