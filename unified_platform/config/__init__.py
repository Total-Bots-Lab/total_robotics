"""
Configuration Layer
==================

Robot configuration, examples, and reward system definitions.
This layer contains all robot-specific configurations and task definitions.
"""

from .universal_config import UniversalRobotConfig, make_robot_config, RobotLibrary
from .reward_system import RewardManager, BaseRewardFunction, create_locomotion_rewards, create_manipulation_rewards
from .robot_examples import get_config_for_robot, get_rewards_for_task, get_obs_for_task

__all__ = [
    # Robot Configuration (Universal)
    'UniversalRobotConfig',
    'make_robot_config',
    'RobotLibrary',
    
    # Reward System
    'RewardManager',
    'BaseRewardFunction',
    'create_locomotion_rewards',
    'create_manipulation_rewards',
    
    # Robot Examples
    'get_config_for_robot',
    'get_rewards_for_task',
    'get_obs_for_task'
]
