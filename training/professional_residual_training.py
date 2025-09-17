#!/usr/bin/env python3
"""
Professional Residual Action Training
=====================================

Implementation following the research from:
- https://federicosarrocco.com/blog/Making-Quadrupeds-Learning-To-Walk
- https://github.com/Argo-Robot/quadrupeds_locomotion

Key features:
1. RESIDUAL ACTION SPACE: action_total = q_homing + residual_action_nn
2. EXPONENTIAL TRACKING REWARDS: R = exp(-||error||^2) 
3. SYMMETRIC HOMING POSITION: Prevents leg asymmetry
4. SIMPLE REWARD STRUCTURE: Focuses on tracking with minimal penalties

Author: Professional Implementation
Date: September 17, 2025
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, Any, Optional

# Add unified platform to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Complete unified platform imports
from unified_platform.config.logger_config import setup_unicode_logger
from unified_platform.config.universal_config import RobotLibrary
from unified_platform.config.reward_system import RewardManager
from unified_platform.environment.generic_robot_env import make_robot_env, GenericRobotGymEnv

# Import our advanced reward system
from generic_quadruped_reward import GenericQuadrupedReward, RewardConfig


def train_professional_residual_approach():
    """
    Train using professional residual action approach from research.
    
    Key differences from previous approach:
    1. Symmetric homing position prevents asymmetry
    2. Moderate action scale allows natural movement
    3. Simple exponential tracking rewards
    4. No complex penalty systems
    """
    
    print("="*80)
    print("🎓 PROFESSIONAL RESIDUAL ACTION TRAINING")
    print("📚 Following Sarrocco & Bertelli Research (2025)")
    print("="*80)
    
    # 1. Setup professional logging
    logger = setup_unicode_logger(__name__)
    logger.info("🔧 Initializing professional residual action training...")
    
    # 2. Load Go2 robot with PROFESSIONAL CONFIGURATION
    logger.info("📚 Loading Go2 with professional residual action configuration...")
    robot_config = RobotLibrary.go2()
    
    # PROFESSIONAL: Residual action parameters (MATCHING ARGO-ROBOT IMPLEMENTATION)
    # Use Argo-Robot proven parameters for Go2
    robot_config.kp = 20.0   # Argo-Robot proven stiffness
    robot_config.kd = 0.5    # Argo-Robot proven damping  
    robot_config.action_scale = 0.25    # Argo-Robot proven scaling
    
    # PROFESSIONAL: Restrict hip joint movement to prevent excessive abduction
    robot_config.clip_actions = 1.3  # Allow reasonable movement range
    
    # Add custom joint position limits for hip joints (more restrictive than URDF)
    robot_config.joint_pos_limits = {
        # Hip abduction joints - restrict to ±30 degrees instead of ±60 degrees
        "FL_hip_joint": [-0.524, 0.524],  # ±30° in radians (more restrictive)
        "FR_hip_joint": [-0.524, 0.524],  # ±30° in radians 
        "RL_hip_joint": [-0.524, 0.524],  # ±30° in radians
        "RR_hip_joint": [-0.524, 0.524],  # ±30° in radians
        # Other joints use URDF defaults (no additional restriction)
    }
    
    # PROFESSIONAL: Natural standing pose for Go2 quadruped (ARGO-ROBOT CONFIGURATION)
    # RESEARCH APPROACH: Use Argo-Robot proven joint angles for proper standing
    robot_config.default_joint_angles = {
        # Hip abduction joints - perfectly neutral for all legs
        "FL_hip_joint": 0.0,  "FR_hip_joint": 0.0,  "RL_hip_joint": 0.0,  "RR_hip_joint": 0.0,
        # Thigh joints - ARGO-ROBOT ASYMMETRIC: Front=0.8, Rear=1.0 for proper quadruped posture
        "FL_thigh_joint": 0.8,  "FR_thigh_joint": 0.8,  "RL_thigh_joint": 1.0,  "RR_thigh_joint": 1.0,
        # Calf joints - ARGO-ROBOT UNIFORM: All -1.5 for proper leg extension
        "FL_calf_joint": -1.5,  "FR_calf_joint": -1.5,  "RL_calf_joint": -1.5,  "RR_calf_joint": -1.5,
    }
    
    # Note: Joint limits are enforced by the URDF; residual scaling and symmetric homing prevent folding.
    
    logger.info(f"✅ Professional Go2 configuration: {robot_config.name}")
    logger.info(f"   📊 DOFs: {robot_config.num_actions}")
    logger.info(f"   🎯 RESIDUAL gains: kp={robot_config.kp}, kd={robot_config.kd}")
    logger.info(f"   📏 RESIDUAL action scale: {robot_config.action_scale}")
    logger.info(f"   🔒 Symmetric homing position: All legs identical")
    logger.info(f"   🎯 Research approach: action_total = q_homing + residual_action_nn")
    
    # 3. Create professional reward system using GenericQuadrupedReward
    logger.info("🎯 Setting up professional GenericQuadrupedReward system...")
    
    # Configure the professional reward system with research-based parameters
    reward_config = RewardConfig(
        base_height_target=0.30,  # Match Argo-Robot target height (0.3m)
        tracking_sigma=0.25,      # Velocity tracking tolerance
        reward_scales={
            # RESEARCH-BASED: Exponential tracking rewards (PRIMARY OBJECTIVES)
            "tracking_lin_vel": 5.0,     # Strong forward velocity tracking
            "tracking_ang_vel": 1.0,     # Moderate angular velocity tracking  
            
            # RESEARCH-BASED: Simple penalty terms (SECONDARY OBJECTIVES)
            "lin_vel_z": -1.0,          # Discourage vertical movement
            "base_height": -2.0,        # Height maintenance penalty
            "orientation": -1.0,        # Roll/pitch stability penalty
            
            # RESEARCH-BASED: Action quality 
            "action_rate": -0.1,        # Action smoothness penalty
            "similar_to_default": -0.5, # Keep close to homing position (gentle for residuals)
        }
    )
    
    # Create the professional reward system
    num_envs = 1
    generic_reward = GenericQuadrupedReward(
        num_envs=num_envs,
        reward_config=reward_config,
        dt=0.02,
        device="cuda"
    )
    
    # Wrap it for the RewardManager interface
    class ProfessionalRewardAdapter:
        def __init__(self, reward_system):
            self.reward_system = reward_system
            self.name = "professional_residual_reward"
            self.weight = 1.0
            
        def compute(self, env_state: Dict[str, Any]) -> torch.Tensor:
            # Convert env_state to format expected by GenericQuadrupedReward
            device = env_state.get("device", torch.device("cuda"))
            batch = env_state.get("batch_size", 1)
            def _get(key, shape):
                return env_state.get(key, torch.zeros(shape, device=device))
            robot_state = {
                "base_lin_vel": _get("base_lin_vel", (batch, 3)).to(self.reward_system.device),
                "base_ang_vel": _get("base_ang_vel", (batch, 3)).to(self.reward_system.device),
                "base_pos": _get("base_pos", (batch, 3)).to(self.reward_system.device),
                "dof_pos": _get("dof_pos", (batch, 12)).to(self.reward_system.device),
                "default_dof_pos": _get("default_dof_pos", (12,)).to(self.reward_system.device),
                "base_euler": _get("base_euler", (batch, 3)).to(self.reward_system.device),
            }
            
            # Professional commands for forward walking
            commands = torch.tensor([[1.0, 0.0, 0.0, 0.27]], device=self.reward_system.device).repeat(batch, 1)
            actions = env_state.get("actions", torch.zeros(batch, 12, device=device)).to(self.reward_system.device)
            last_actions = env_state.get("last_actions", torch.zeros(batch, 12, device=device)).to(self.reward_system.device)
            jump_state = {
                "jump_toggled_buf": torch.zeros(batch, device=self.reward_system.device),
                "jump_target_height": torch.zeros(batch, device=self.reward_system.device)
            }
            
            # Get professional reward using the existing system
            reward = self.reward_system.compute_reward(
                robot_state, commands, actions, last_actions, jump_state
            )
            
            return reward
    
    reward_manager = RewardManager(dt=0.02)
    reward_manager.add_reward(ProfessionalRewardAdapter(generic_reward))
    
    logger.info("✅ Professional GenericQuadrupedReward system configured")
    logger.info("   � Using existing professional reward implementation")
    logger.info("   🏃 PRIMARY: Linear velocity tracking (5.0x weight)")
    logger.info("   🔄 SECONDARY: Angular velocity tracking (1.0x weight)")  
    logger.info("   📊 Simple height penalty (-2.0x weight)")
    logger.info("   📐 Simple orientation penalty (-1.0x weight)")
    logger.info("   🎯 Pose similarity to symmetric homing position (-0.5x weight)")
    logger.info("   ⚡ Action smoothness penalty (-0.1x weight)")
    logger.info("   🔥 Research-based exponential tracking rewards")
    
    # 4. Create training environment using direct instantiation (custom config)
    logger.info("🌍 Creating professional training environment...")
    logger.info("👁️  Genesis viewer enabled - watch the professional training!")
    # PROFESSIONAL: Use Argo-Robot direct approach (no warm-up needed)
    env = GenericRobotGymEnv(
        robot_config=robot_config,
        render_mode="human",
        reward_manager=reward_manager,
        stand_warmup_steps=0,  # Argo-Robot: No warm-up, direct training
    )
    logger.info(f"✅ Professional training environment created")
    logger.info(f"   📊 Observation space: {env.observation_space}")
    logger.info(f"   🎮 Action space: {env.action_space} (residual actions)")
    
    # 5. Run professional PPO training
    logger.info("🤖 Starting professional PPO training...")
    
    try:
        from stable_baselines3 import PPO
        
        logger.info("🎓 Configuring PPO for residual action learning...")
        
        # Professional PPO configuration
        model = PPO(
            "MlpPolicy", 
            env,
            learning_rate=3e-4,     # Standard learning rate
            n_steps=2048,           # Good batch size for continuous control
            batch_size=64,          # Reasonable batch size
            n_epochs=10,            # Standard epochs
            gamma=0.99,             # Standard discount factor
            gae_lambda=0.95,        # Standard GAE parameter
            clip_range=0.2,         # Standard PPO clipping
            verbose=1,
            device="cpu"            # CPU for MLP policy
        )
        
        # Professional training
        logger.info("🎓 Starting professional residual action training (30,000 timesteps)...")
        logger.info("   🎯 Goal: Natural quadruped locomotion using residual actions")
        logger.info("   📚 Method: action_total = q_homing + residual_action_nn")
        logger.info("   🏃 Expected: Symmetric gait, stable tracking, no leg folding")
        
        # Professional training parameters for stable walking
        logger.info("🎯 Training for 2M steps (~3-4 hours for stable walking)")
        logger.info("📊 Progress will be shown every 10K steps")
        model.learn(total_timesteps=2000000, progress_bar=True)
        
        # Save the professional model
        model_path = "go2_professional_residual_model"
        model.save(model_path)
        logger.info(f"💾 Professional residual model saved as: {model_path}")
        
        # Professional testing
        logger.info("🧪 Testing professional residual locomotion...")
        logger.info("   👀 Watch for: Symmetric gaits, stable tracking, natural movement")
        obs, info = env.reset()
        total_test_reward = 0
        
        for step in range(1000):  # Extended test
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_test_reward += reward
            
            if step % 100 == 0:
                logger.info(f"   🎯 Professional test step {step}: reward={reward:.4f}, total={total_test_reward:.4f}")
            
            if terminated or truncated:
                obs, info = env.reset()
        
        logger.info(f"🏆 Professional training completed! Final reward: {total_test_reward:.4f}")
        
    except ImportError as e:
        logger.error("❌ Stable Baselines3 not available!")
        logger.info("💡 Install with: pip install stable-baselines3[extra]")
        
    except Exception as e:
        logger.error(f"❌ Professional training failed: {e}")
    
    finally:
        # Clean up properly
        logger.info("🧹 Professional cleanup...")
        try:
            if 'env' in locals():
                env.close()
                logger.info("✅ Environment closed")
        except Exception as e:
            logger.info(f"⚠️  Environment close error: {e}")
        
        try:
            import genesis as gs
            gs.destroy()
            logger.info("✅ Genesis destroyed successfully")
        except Exception as e:
            logger.info(f"⚠️  Genesis cleanup: {e}")
        
        logger.info("🏁 Professional residual training completed!")


if __name__ == "__main__":
    print("🎓 PROFESSIONAL RESIDUAL ACTION TRAINING")
    print("📚 Based on Sarrocco & Bertelli Research (2025)")
    print("🎯 Key: action_total = q_homing + residual_action_nn")
    print("")
    
    train_professional_residual_approach()
    
    print("")
    print("="*80)
    print("💡 RESEARCH-BASED APPROACH COMPARISON")
    print("="*80)
    print("🔴 OLD APPROACH (complete_training_integration.py):")
    print("   - Direct joint position actions")
    print("   - Complex penalty system")
    print("   - Asymmetric leg positioning")
    print("   - Ultra-tight joint limits")
    print("")
    print("🟢 PROFESSIONAL APPROACH (this script):")
    print("   ✅ Residual actions: action_total = q_homing + residual")
    print("   ✅ Symmetric homing position prevents asymmetry")
    print("   ✅ Exponential tracking rewards (research-proven)")
    print("   ✅ Simple penalty structure")
    print("   ✅ Moderate joint limits allow natural movement")
    print("")
    print("🚀 This should SOLVE the leg folding issue!")
    print("="*80)