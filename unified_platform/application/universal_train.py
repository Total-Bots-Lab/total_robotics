#!/usr/bin/env python3
"""
Universal Robot Training Script
==============================

Works with any robot URDF file. Users just need to specify robot configuration.
"""

import argparse
import sys
import os
import numpy as np
from typing import Optional

# Add the parent directory (test) to Python path so we can import unified_platform
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# Import our generic system
from unified_platform.environment.generic_robot_env import make_robot_env, make_custom_env, GenericRobotGymEnv
from unified_platform.config.universal_config import make_robot_config, RobotLibrary
from unified_platform.config.reward_system import create_custom_rewards


def test_predefined_robot(robot_name: str):
    """Test a predefined robot configuration."""
    print(f"Testing {robot_name} robot...")
    
    try:
        env = make_robot_env(robot_name, render_mode="human")
        
        obs, info = env.reset()
        print(f"Observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")
        
        total_reward = 0
        for step in range(200):
            # Random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 50 == 0:
                print(f"Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
            
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                obs, info = env.reset()
                total_reward = 0
                
        env.close()
        print(f"✅ {robot_name} test completed successfully!")
        
    except Exception as e:
        print(f"❌ {robot_name} test failed: {e}")


def test_custom_robot():
    """Test with a custom robot configuration."""
    print("Testing custom robot configuration...")
    
    # Example: Custom 6-DOF arm
    try:
        env = make_custom_env(
            urdf_path="urdf/custom_arm/arm.urdf",  # User's URDF
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            default_joint_angles={
                "joint1": 0.0, "joint2": -0.5, "joint3": 0.0,
                "joint4": -1.5, "joint5": 0.0, "joint6": 0.0
            },
            base_init_pos=[0.0, 0.0, 0.5],
            kp=100.0,  # Stiffer control
            kd=5.0,
            action_scale=0.1,
            render_mode="human"
        )
        
        obs, info = env.reset()
        print(f"Custom robot observation shape: {obs.shape}")
        
        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 25 == 0:
                print(f"Step {step}: reward={reward:.3f}")
            
            if terminated or truncated:
                obs, info = env.reset()
        
        env.close()
        print("✅ Custom robot test completed!")
        
    except Exception as e:
        print(f"❌ Custom robot test failed: {e}")


def demo_advanced_usage():
    """Demonstrate advanced customization."""
    print("Demonstrating advanced customization...")
    
    # 1. Custom robot config
    robot_config = RobotLibrary.custom(
        name="my_robot",
        urdf_path="urdf/my_robot/robot.urdf",
        joint_names=["j1", "j2", "j3", "j4"],
        default_joint_angles={"j1": 0.0, "j2": 0.5, "j3": -0.5, "j4": 0.0},
        base_init_pos=[0.0, 0.0, 0.3],
        kp=50.0,
        kd=2.0,
    )
    
    # 2. Custom reward configuration
    reward_configs = [
        {"type": "position_tracking", "weight": 10.0, "params": {"sigma": 0.05}},
        {"type": "action_smoothness", "weight": -0.1},
        {"type": "joint_regularization", "weight": -0.01},
    ]
    reward_manager = create_custom_rewards(reward_configs)
    
    # 3. Custom observation components
    obs_components = [
        {"name": "joint_pos", "size": 4, "scale": 1.0},
        {"name": "joint_vel", "size": 4, "scale": 0.1},
        {"name": "actions", "size": 4, "scale": 1.0},
    ]
    
    try:
        # Create fully customized environment
        env = GenericRobotGymEnv(
            robot_config=robot_config,
            reward_manager=reward_manager,
            obs_components=obs_components,
            render_mode=None
        )
        
        print(f"Advanced customization - Obs shape: {env.observation_space.shape}")
        print("✅ Advanced customization works!")
        
    except Exception as e:
        print(f"❌ Advanced customization failed (expected if URDF doesn't exist): {e}")


def train_with_sb3(robot_name: str):
    """Train any robot with Stable Baselines3."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        print(f"Training {robot_name} with Stable Baselines3...")
        
        # Create environment
        def make_env():
            return make_robot_env(robot_name, render_mode=None)
        
        env = DummyVecEnv([make_env])
        
        # Create PPO model
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
        
        # Train
        print("Training for 10000 timesteps...")
        model.learn(total_timesteps=10000)
        
        # Test
        print("Testing trained model...")
        obs = env.reset()
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
        
        env.close()
        print(f"✅ {robot_name} training completed!")
        
    except ImportError:
        print("❌ Stable Baselines3 not available. Install with: pip install stable-baselines3")
    except Exception as e:
        print(f"❌ Training failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Universal Robot Environment Testing")
    parser.add_argument("--robot", choices=["go2", "franka", "anymal", "custom"], 
                       default="go2", help="Robot to test")
    parser.add_argument("--mode", choices=["test", "train", "demo"], 
                       default="test", help="Mode to run")
    
    args = parser.parse_args()
    
    print("🚀 Universal Robot Environment System")
    print("=" * 50)
    
    if args.mode == "test":
        if args.robot == "custom":
            test_custom_robot()
        else:
            test_predefined_robot(args.robot)
    
    elif args.mode == "train":
        if args.robot == "custom":
            print("Training not available for custom robots in this demo")
        else:
            train_with_sb3(args.robot)
    
    elif args.mode == "demo":
        demo_advanced_usage()
    
    print("\n🎯 How to use for YOUR robot:")
    print("1. Place your URDF file in the Genesis urdf directory")
    print("2. Create robot config with joint names and default angles")  
    print("3. Optionally customize rewards and observations")
    print("4. Use make_custom_env() or GenericRobotGymEnv()")
    print("\nMinimal example:")
    print("env = make_custom_env('path/to/robot.urdf', ['joint1', 'joint2', ...])")


if __name__ == "__main__":
    main()
