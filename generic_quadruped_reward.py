"""
Generic Quadruped Reward System

A flexible and configurable reward system for quadruped locomotion that can be adapted
to different robot platforms (Go2, Spot, ANYmal, Mini Cheetah, etc.).

Key Features:
- Modular reward components
- Robot-agnostic design
- Configurable scaling and parameters
- Support for walking, height control, and jumping behaviors
- Easy integration with different simulation environments

Author: Sayantan Brahma
Date: September 10, 2025
"""

import torch
import math
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration class for reward system parameters"""
    
    # Tracking parameters
    tracking_sigma: float = 0.25
    base_height_target: float = 0.3
    feet_height_target: float = 0.075
    
    # Jump parameters
    jump_upward_velocity: float = 1.2
    jump_reward_steps: int = 50
    
    # Reward component scales
    reward_scales: Dict[str, float] = None
    
    # Termination thresholds
    max_roll_pitch: float = 10.0  # degrees
    
    def __post_init__(self):
        if self.reward_scales is None:
            self.reward_scales = {
                "tracking_lin_vel": 1.0,
                "tracking_ang_vel": 0.2,
                "lin_vel_z": -1.0,
                "base_height": -50.0,
                "action_rate": -0.005,
                "similar_to_default": -0.1,
                "jump_height_tracking": 0.5,
                "jump_height_achievement": 10.0,
                "jump_speed": 1.0,
                "jump_landing": 0.08,
            }


class GenericQuadrupedReward:
    """
    Generic reward system for quadruped locomotion.
    
    This class provides a flexible reward framework that can be adapted to different
    quadruped robots by configuring robot-specific parameters and reward weights.
    """
    
    def __init__(self, 
                 num_envs: int,
                 reward_config: RewardConfig,
                 dt: float = 0.02,
                 device: str = "cuda"):
        """
        Initialize the reward system.
        
        Args:
            num_envs: Number of parallel environments
            reward_config: Configuration object with reward parameters
            dt: Simulation timestep
            device: Torch device (cuda/cpu)
        """
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.dt = dt
        self.config = reward_config
        
        # Scale rewards by timestep
        self.reward_scales = {}
        for name, scale in self.config.reward_scales.items():
            self.reward_scales[name] = scale * self.dt
        
        # Initialize reward tracking
        self.reward_functions = {}
        self.episode_sums = {}
        
        # Register all available reward functions
        self._register_reward_functions()
        
        # Initialize buffers for episode tracking
        for name in self.reward_scales.keys():
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device)
    
    def _register_reward_functions(self):
        """Register all available reward functions"""
        self.reward_functions = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "lin_vel_z": self._reward_lin_vel_z,
            "action_rate": self._reward_action_rate,
            "similar_to_default": self._reward_similar_to_default,
            "base_height": self._reward_base_height,
            "jump_height_tracking": self._reward_jump_height_tracking,
            "jump_height_achievement": self._reward_jump_height_achievement,
            "jump_speed": self._reward_jump_speed,
            "jump_landing": self._reward_jump_landing,
        }
    
    def compute_reward(self, 
                      robot_state: Dict[str, torch.Tensor],
                      commands: torch.Tensor,
                      actions: torch.Tensor,
                      last_actions: torch.Tensor,
                      jump_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute total reward for all environments.
        
        Args:
            robot_state: Dictionary containing robot state information
                - base_lin_vel: (num_envs, 3) base linear velocity
                - base_ang_vel: (num_envs, 3) base angular velocity
                - base_pos: (num_envs, 3) base position
                - dof_pos: (num_envs, num_dofs) joint positions
                - default_dof_pos: (num_dofs,) default joint positions
            commands: (num_envs, num_commands) command vector
            actions: (num_envs, num_actions) current actions
            last_actions: (num_envs, num_actions) previous actions
            jump_state: Dictionary containing jump-related state
                - jump_toggled_buf: (num_envs,) jump activation buffer
                - jump_target_height: (num_envs,) target jump heights
        
        Returns:
            torch.Tensor: (num_envs,) total reward for each environment
        """
        # Store state for reward computation
        self.robot_state = robot_state
        self.commands = commands
        self.actions = actions
        self.last_actions = last_actions
        self.jump_state = jump_state
        
        # Compute individual reward components
        total_reward = torch.zeros((self.num_envs,), device=self.device)
        
        for name, reward_func in self.reward_functions.items():
            if name in self.reward_scales:
                reward = reward_func() * self.reward_scales[name]
                total_reward += reward
                self.episode_sums[name] += reward
        
        return total_reward
    
    def reset_episode_sums(self, env_ids: torch.Tensor) -> Dict[str, float]:
        """
        Reset episode sums for specified environments and return averages.
        
        Args:
            env_ids: Tensor of environment indices to reset
            
        Returns:
            Dictionary of average rewards per component
        """
        episode_rewards = {}
        for key in self.episode_sums.keys():
            if len(env_ids) > 0:
                episode_rewards["rew_" + key] = torch.mean(
                    self.episode_sums[key][env_ids]
                ).item() / (self.dt * 1000)  # Normalize by episode length
                self.episode_sums[key][env_ids] = 0.0
        
        return episode_rewards
    
    # ==================== REWARD FUNCTIONS ====================
    
    def _reward_tracking_lin_vel(self) -> torch.Tensor:
        """Reward for tracking linear velocity commands (xy axes) - Research-based exponential"""
        # Handle both 2D and 4D command formats
        if self.commands.shape[1] >= 2:
            target_vel = self.commands[:, :2]  # [vx, vy]
        else:
            # Fallback for single command format
            target_vel = torch.stack([self.commands[:, 0], torch.zeros_like(self.commands[:, 0])], dim=1)
        
        actual_vel = self.robot_state["base_lin_vel"][:, :2]
        
        # Research approach: R = exp(-||v_ref - v_actual||^2 / sigma)
        vel_error = torch.norm(target_vel - actual_vel, dim=1)**2
        return torch.exp(-vel_error / self.config.tracking_sigma**2)
    
    def _reward_tracking_ang_vel(self) -> torch.Tensor:
        """Reward for tracking angular velocity commands (yaw) - Research-based exponential"""
        # Handle both 3D and 4D command formats
        if self.commands.shape[1] >= 3:
            target_ang_vel = self.commands[:, 2]  # yaw velocity
        else:
            target_ang_vel = torch.zeros_like(self.commands[:, 0])  # no turning
        
        actual_ang_vel = self.robot_state["base_ang_vel"][:, 2]  # yaw velocity
        
        # Research approach: R = exp(-(w_ref - w_actual)^2 / sigma)
        ang_vel_error = (target_ang_vel - actual_ang_vel)**2
        return torch.exp(-ang_vel_error / self.config.tracking_sigma**2)
    
    def _reward_lin_vel_z(self) -> torch.Tensor:
        """Penalty for vertical velocity (disabled during jumping)"""
        active_mask = (self.jump_state["jump_toggled_buf"] < 0.01).float()
        return active_mask * torch.square(self.robot_state["base_lin_vel"][:, 2])
    
    def _reward_action_rate(self) -> torch.Tensor:
        """Penalty for rapid changes in actions (disabled during jumping)"""
        active_mask = (self.jump_state["jump_toggled_buf"] < 0.01).float()
        return active_mask * torch.sum(
            torch.square(self.last_actions - self.actions), dim=1
        )
    
    def _reward_similar_to_default(self) -> torch.Tensor:
        """Penalty for joint positions far from default (disabled during jumping)"""
        active_mask = (self.jump_state["jump_toggled_buf"] < 0.01).float()
        return active_mask * torch.sum(
            torch.abs(self.robot_state["dof_pos"] - self.robot_state["default_dof_pos"]), 
            dim=1
        )
    
    def _reward_base_height(self) -> torch.Tensor:
        """Reward for maintaining target base height - Research-based exponential"""
        active_mask = (self.jump_state["jump_toggled_buf"] < 0.01).float()
        # Use commanded height if available, otherwise use target height
        target_height = self.commands[:, 3] if self.commands.shape[1] > 3 else self.config.base_height_target
        height_error = (self.robot_state["base_pos"][:, 2] - target_height)**2
        return active_mask * torch.exp(-height_error / self.config.tracking_sigma**2)
    
    def _reward_jump_height_tracking(self) -> torch.Tensor:
        """Continuous reward for minimizing distance to target height during peak phase"""
        mask = ((self.jump_state["jump_toggled_buf"] >= 0.3 * self.config.jump_reward_steps) & 
                (self.jump_state["jump_toggled_buf"] < 0.6 * self.config.jump_reward_steps))
        target_height = self.jump_state["jump_target_height"]
        height_diff = torch.exp(-torch.square(self.robot_state["base_pos"][:, 2] - target_height))
        return mask.float() * height_diff
    
    def _reward_jump_height_achievement(self) -> torch.Tensor:
        """Binary reward for reaching close to target height during peak phase"""
        mask = ((self.jump_state["jump_toggled_buf"] >= 0.3 * self.config.jump_reward_steps) & 
                (self.jump_state["jump_toggled_buf"] < 0.6 * self.config.jump_reward_steps))
        target_height = self.jump_state["jump_target_height"]
        binary_bonus = (torch.abs(self.robot_state["base_pos"][:, 2] - target_height) < 0.2).float()
        return mask.float() * binary_bonus
    
    def _reward_jump_speed(self) -> torch.Tensor:
        """Reward for upward velocity during peak phase"""
        mask = ((self.jump_state["jump_toggled_buf"] >= 0.3 * self.config.jump_reward_steps) & 
                (self.jump_state["jump_toggled_buf"] < 0.6 * self.config.jump_reward_steps))
        return mask.float() * torch.exp(self.robot_state["base_lin_vel"][:, 2]) * 0.2
    
    def _reward_jump_landing(self) -> torch.Tensor:
        """Penalty for deviation from base height during landing"""
        mask = (self.jump_state["jump_toggled_buf"] >= 0.6 * self.config.jump_reward_steps)
        height_error = -torch.square(
            self.robot_state["base_pos"][:, 2] - self.config.base_height_target
        )
        return mask.float() * height_error
    
    # ==================== UTILITY METHODS ====================
    
    def add_custom_reward(self, name: str, reward_func: Callable, scale: float):
        """
        Add a custom reward function.
        
        Args:
            name: Name of the reward component
            reward_func: Function that returns reward tensor
            scale: Scaling factor for the reward
        """
        self.reward_functions[name] = reward_func
        self.reward_scales[name] = scale * self.dt
        self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device)
    
    def remove_reward(self, name: str):
        """Remove a reward component"""
        if name in self.reward_functions:
            del self.reward_functions[name]
        if name in self.reward_scales:
            del self.reward_scales[name]
        if name in self.episode_sums:
            del self.episode_sums[name]
    
    def update_reward_scale(self, name: str, new_scale: float):
        """Update the scale of a reward component"""
        if name in self.reward_scales:
            self.reward_scales[name] = new_scale * self.dt
    
    def get_reward_info(self) -> Dict[str, Any]:
        """Get information about current reward configuration"""
        return {
            "active_rewards": list(self.reward_functions.keys()),
            "reward_scales": self.reward_scales.copy(),
            "config": self.config
        }
    
    def check_termination(self, robot_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Check termination conditions based on robot orientation.
        
        Args:
            robot_state: Dictionary containing robot state with 'base_euler' key
            
        Returns:
            torch.Tensor: Boolean tensor indicating which environments should terminate
        """
        if "base_euler" not in robot_state:
            return torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        
        max_angle_rad = math.radians(self.config.max_roll_pitch)
        roll_pitch_violation = (
            (torch.abs(robot_state["base_euler"][:, 0]) > max_angle_rad) |  # roll
            (torch.abs(robot_state["base_euler"][:, 1]) > max_angle_rad)    # pitch
        )
        
        return roll_pitch_violation


class RewardProfiler:
    """Utility class for profiling and analyzing reward components"""
    
    def __init__(self, reward_system: GenericQuadrupedReward):
        self.reward_system = reward_system
        self.profile_data = {}
    
    def profile_rewards(self, 
                       robot_state: Dict[str, torch.Tensor],
                       commands: torch.Tensor,
                       actions: torch.Tensor,
                       last_actions: torch.Tensor,
                       jump_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Profile individual reward components.
        
        Returns:
            Dictionary with reward component values
        """
        # Store state
        self.reward_system.robot_state = robot_state
        self.reward_system.commands = commands
        self.reward_system.actions = actions
        self.reward_system.last_actions = last_actions
        self.reward_system.jump_state = jump_state
        
        profile = {}
        for name, reward_func in self.reward_system.reward_functions.items():
            if name in self.reward_system.reward_scales:
                raw_reward = reward_func()
                scaled_reward = raw_reward * self.reward_system.reward_scales[name]
                profile[name] = {
                    "raw_mean": torch.mean(raw_reward).item(),
                    "raw_std": torch.std(raw_reward).item(),
                    "scaled_mean": torch.mean(scaled_reward).item(),
                    "scaled_std": torch.std(scaled_reward).item(),
                    "scale": self.reward_system.reward_scales[name]
                }
        
        return profile
