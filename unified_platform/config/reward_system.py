"""
Generic Reward System
====================

Modular reward system that works with any robot configuration.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Any
import torch
import numpy as np


class BaseRewardFunction(ABC):
    """Base class for reward functions."""
    
    def __init__(self, weight: float = 1.0, name: str = ""):
        self.weight = weight
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def compute(self, env_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute reward given environment state."""
        pass


class VelocityTrackingReward(BaseRewardFunction):
    """Reward for tracking velocity commands (locomotion robots)."""
    
    def __init__(self, weight: float = 1.0, sigma: float = 0.25):
        super().__init__(weight, "velocity_tracking")
        self.sigma = sigma
    
    def compute(self, env_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "commands" not in env_state or "base_lin_vel" not in env_state:
            return torch.zeros(env_state["batch_size"], device=env_state["device"])
        
        commands = env_state["commands"]
        base_lin_vel = env_state["base_lin_vel"]
        
        # Track linear velocity (xy)
        lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.sigma)


class PositionTrackingReward(BaseRewardFunction):
    """Reward for tracking position targets (manipulation robots)."""
    
    def __init__(self, weight: float = 1.0, sigma: float = 0.1):
        super().__init__(weight, "position_tracking")
        self.sigma = sigma
    
    def compute(self, env_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "target_pos" not in env_state or "end_effector_pos" not in env_state:
            return torch.zeros(env_state["batch_size"], device=env_state["device"])
        
        target_pos = env_state["target_pos"]
        ee_pos = env_state["end_effector_pos"]
        
        pos_error = torch.norm(target_pos - ee_pos, dim=1)
        return torch.exp(-pos_error / self.sigma)


class ActionSmoothnessReward(BaseRewardFunction):
    """Penalize rapid changes in actions (universal)."""
    
    def __init__(self, weight: float = -0.01):
        super().__init__(weight, "action_smoothness")
    
    def compute(self, env_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        actions = env_state["actions"]
        last_actions = env_state["last_actions"]
        
        action_diff = torch.sum(torch.square(actions - last_actions), dim=1)
        return action_diff


class JointRegularizationReward(BaseRewardFunction):
    """Penalize deviation from default joint positions (universal)."""
    
    def __init__(self, weight: float = -0.1):
        super().__init__(weight, "joint_regularization")
    
    def compute(self, env_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        dof_pos = env_state["dof_pos"]
        default_dof_pos = env_state["default_dof_pos"]
        
        joint_deviation = torch.sum(torch.abs(dof_pos - default_dof_pos), dim=1)
        return joint_deviation


class StabilityReward(BaseRewardFunction):
    """Reward for maintaining stable base orientation (locomotion robots)."""
    
    def __init__(self, weight: float = -1.0):
        super().__init__(weight, "stability")
    
    def compute(self, env_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "base_euler" not in env_state:
            return torch.zeros(env_state["batch_size"], device=env_state["device"])
        
        base_euler = env_state["base_euler"]
        
        # Penalize roll and pitch deviations
        roll_penalty = torch.square(base_euler[:, 0])  # Roll
        pitch_penalty = torch.square(base_euler[:, 1])  # Pitch
        
        return roll_penalty + pitch_penalty


class CollisionAvoidanceReward(BaseRewardFunction):
    """Penalize collisions (manipulation robots)."""
    
    def __init__(self, weight: float = -10.0):
        super().__init__(weight, "collision_avoidance")
    
    def compute(self, env_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "collision" not in env_state:
            return torch.zeros(env_state["batch_size"], device=env_state["device"])
        
        # collision is binary tensor: 1 if collision, 0 otherwise
        return env_state["collision"].float()


class RewardManager:
    """Manages multiple reward functions."""
    
    def __init__(self, dt: float = 0.02):
        self.reward_functions: List[BaseRewardFunction] = []
        self.dt = dt
        self.episode_sums: Dict[str, torch.Tensor] = {}
    
    def add_reward(self, reward_func: BaseRewardFunction):
        """Add a reward function."""
        # Scale weight by dt for consistent scaling across different frequencies
        reward_func.weight *= self.dt
        self.reward_functions.append(reward_func)
    
    def reset_episode_sums(self, num_envs: int, device: torch.device):
        """Reset episode sum tracking."""
        self.episode_sums = {
            func.name: torch.zeros(num_envs, device=device, dtype=torch.float32)
            for func in self.reward_functions
        }
    
    def compute_total_reward(self, env_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute total weighted reward."""
        total_reward = torch.zeros(env_state["batch_size"], device=env_state["device"])
        
        for reward_func in self.reward_functions:
            reward_value = reward_func.compute(env_state)
            weighted_reward = reward_value * reward_func.weight
            total_reward += weighted_reward
            
            # Track episode sums
            if reward_func.name in self.episode_sums:
                self.episode_sums[reward_func.name] += weighted_reward
        
        return total_reward
    
    def get_reward_breakdown(self) -> Dict[str, torch.Tensor]:
        """Get breakdown of individual reward components."""
        return self.episode_sums.copy()


def create_locomotion_rewards(dt: float = 0.02) -> RewardManager:
    """Create reward setup for locomotion robots (Go2, ANYmal, etc)."""
    manager = RewardManager(dt)
    
    # Locomotion-specific rewards
    manager.add_reward(VelocityTrackingReward(weight=1.0))
    manager.add_reward(StabilityReward(weight=-1.0))
    manager.add_reward(ActionSmoothnessReward(weight=-0.01))
    manager.add_reward(JointRegularizationReward(weight=-0.1))
    
    return manager


def create_manipulation_rewards(dt: float = 0.02) -> RewardManager:
    """Create reward setup for manipulation robots (Franka, UR5, etc)."""
    manager = RewardManager(dt)
    
    # Manipulation-specific rewards
    manager.add_reward(PositionTrackingReward(weight=10.0))
    manager.add_reward(CollisionAvoidanceReward(weight=-10.0))
    manager.add_reward(ActionSmoothnessReward(weight=-0.1))
    manager.add_reward(JointRegularizationReward(weight=-0.01))
    
    return manager


def create_custom_rewards(reward_configs: List[Dict[str, Any]], dt: float = 0.02) -> RewardManager:
    """Create custom reward setup from configuration."""
    manager = RewardManager(dt)
    
    # Map of available reward functions
    reward_classes = {
        "velocity_tracking": VelocityTrackingReward,
        "position_tracking": PositionTrackingReward,
        "action_smoothness": ActionSmoothnessReward,
        "joint_regularization": JointRegularizationReward,
        "stability": StabilityReward,
        "collision_avoidance": CollisionAvoidanceReward,
    }
    
    for config in reward_configs:
        reward_type = config["type"]
        if reward_type not in reward_classes:
            available = ", ".join(reward_classes.keys())
            raise ValueError(f"Unknown reward type '{reward_type}'. Available: {available}")
        
        # Extract parameters
        params = config.get("params", {})
        weight = config.get("weight", 1.0)
        
        # Create reward function
        reward_func = reward_classes[reward_type](weight=weight, **params)
        manager.add_reward(reward_func)
    
    return manager
