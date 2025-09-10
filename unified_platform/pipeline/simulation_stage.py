"""
Simulation Stage Implementation
==============================

Implements the complete simulation pipeline from the workflow:
1. Integrate Physics Engine (Genesis AI, Isaac Sim, Gazebo, etc.)
2. Setup the Simulation Environment  
3. Import the Robot
4. Integrate the Controller in the Robot
5. Define Task and Setup Reward Function
6. Run Simulation in Local System/Cloud
7. Train Controller
8. Export Firmware for the Robot
9. Automated Report Generation

This is the core simulation component of the robotics development pipeline.
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch

# Import our unified platform components
from unified_platform.config.universal_config import (
    UniversalPlatformConfig, UniversalRobotConfig, RobotLibrary, make_robot_config
)
from unified_platform.config.reward_system import RewardManager, create_locomotion_rewards, create_manipulation_rewards
from unified_platform.environment.generic_robot_env import GenericRobotGymEnv, make_robot_env
from unified_platform.config.logger_config import create_simulation_logger, configure_genesis_logging, Emojis
from unified_platform.config.universal_config import UniversalPlatformConfig

try:
    import genesis as gs
    GENESIS_AVAILABLE = True
except ImportError:
    GENESIS_AVAILABLE = False
    print("Genesis not available - simulation features limited")


class SimulationStage:
    """
    Complete simulation stage implementation covering all workflow components.
    Now uses UniversalPlatformConfig instead of hardcoded values.
    """
    
    def __init__(self, config: UniversalPlatformConfig = None):
        """
        Initialize simulation stage with universal configuration.
        
        Args:
            config: Universal platform configuration
        """
        self.config = config or UniversalPlatformConfig()
        self.robot_config: Optional[UniversalRobotConfig] = None
        self.reward_manager: Optional[RewardManager] = None
        self.environment: Optional[GenericRobotGymEnv] = None
        self.trained_model = None
        
        # Configure Genesis logging before any Genesis operations
        configure_genesis_logging()
        
        # Setup Unicode-safe logging
        self.logger = create_simulation_logger(self.config.output.log_dir)
        
        # Initialize physics engine
        self.setup_physics_engine()
        
    def setup_physics_engine(self):
        """Step 1: Integrate Physics Engine (Genesis AI, Isaac Sim, Gazebo, etc.)"""
        physics = self.config.physics
        self.logger.info(f"{Emojis.GEAR} Setting up physics engine: {physics.physics_engine}")
        
        if physics.physics_engine == "genesis":
            if not GENESIS_AVAILABLE:
                raise ImportError("Genesis not available. Please install Genesis.")
            
            # Check if Genesis is already initialized
            try:
                # Initialize Genesis with configured backend
                backend = gs.gpu if physics.backend == "gpu" else gs.cpu
                gs.init(backend=backend)
                self.logger.info(f"{Emojis.SUCCESS} Genesis physics engine initialized with {physics.backend.upper()} backend")
            except Exception as e:
                if "already initialized" in str(e):
                    self.logger.info(f"{Emojis.INFO} Genesis already initialized - continuing")
                else:
                    raise e
            
        elif physics.physics_engine == "isaac":
            # TODO: Add Isaac Sim integration
            raise NotImplementedError("Isaac Sim integration not yet implemented")
            
        elif physics.physics_engine == "gazebo":
            # TODO: Add Gazebo integration  
            raise NotImplementedError("Gazebo integration not yet implemented")
            
        else:
            raise ValueError(f"Unsupported physics engine: {physics.physics_engine}")
    
    def setup_simulation_environment(self, terrain_type: str = "flat", **terrain_params):
        """Step 2: Setup the Simulation Environment"""
        self.logger.info("Setting up simulation environment...")
        
        # This will be called when we create the environment
        # For now, we store the configuration
        self.terrain_config = {
            "type": terrain_type,
            "params": terrain_params
        }
        
        self.logger.info(f"✅ Environment configured: {terrain_type} terrain")
    
    def import_robot(self, robot_config: Union[str, UniversalRobotConfig, dict]):
        """Step 3: Import the Robot"""
        self.logger.info("Importing robot...")
        
        if isinstance(robot_config, str):
            # Load predefined robot
            self.robot_config = make_robot_config(robot_config)
            self.logger.info(f"✅ Loaded predefined robot: {robot_config}")
            
        elif isinstance(robot_config, dict):
            # Create custom robot from dict
            self.robot_config = RobotLibrary.custom(**robot_config)
            self.logger.info(f"✅ Created custom robot: {robot_config.get('name', 'unnamed')}")
            
        elif isinstance(robot_config, UniversalRobotConfig):
            # Use provided robot config
            self.robot_config = robot_config
            self.logger.info(f"✅ Using provided robot config: {robot_config.name}")
            
        else:
            raise ValueError("Invalid robot_config type")
    
    def integrate_controller(self, controller_config: dict = None):
        """Step 4: Integrate the Controller in the Robot"""
        self.logger.info("Integrating controller...")
        
        if not self.robot_config:
            raise ValueError("Robot must be imported before integrating controller")
        
        # Controller configuration is stored in robot_config
        # Additional controller parameters can be provided
        if controller_config:
            for key, value in controller_config.items():
                if hasattr(self.robot_config, key):
                    setattr(self.robot_config, key, value)
        
        self.logger.info(f"✅ Controller integrated - kp: {self.robot_config.kp}, kd: {self.robot_config.kd}")
    
    def define_task_and_rewards(self, task_type: str = None):
        """Step 5: Define Task and Setup Reward Function"""
        task_type = task_type or self.config.task.task_type
        self.logger.info(f"{Emojis.TARGET} Defining task: {task_type}")
        
        if task_type == "locomotion":
            self.reward_manager = create_locomotion_rewards(dt=self.config.physics.dt)
        elif task_type == "manipulation":
            self.reward_manager = create_manipulation_rewards(dt=self.config.physics.dt)
        elif task_type == "custom":
            from unified_platform.config.reward_system import create_custom_rewards
            # Use reward components from config
            reward_configs = self.config.reward.reward_components
            self.reward_manager = create_custom_rewards(reward_configs, dt=self.config.physics.dt)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        self.logger.info(f"{Emojis.SUCCESS} Task defined with {len(self.reward_manager.reward_functions)} reward components")
    
    def run_simulation(self, steps: int = None):
        """Step 6: Run Simulation in Local System/Cloud"""
        self.logger.info(f"{Emojis.ROCKET} Running simulation...")
        
        if not all([self.robot_config, self.reward_manager]):
            raise ValueError("Robot and rewards must be configured before running simulation")
        
        # Create environment using config
        render_mode = None
        if self.config.rendering.enable_rendering:
            render_mode = self.config.rendering.render_mode
            # Force human mode if show_viewer is True
            if self.config.rendering.show_viewer:
                render_mode = "human"
        
        self.environment = GenericRobotGymEnv(
            robot_config=self.robot_config,
            reward_manager=self.reward_manager,
            num_envs=self.config.task.num_parallel_envs,
            render_mode=render_mode
        )
        
        steps = steps or self.config.task.max_episode_steps
        
        # Run simulation
        obs, info = self.environment.reset()
        total_reward = 0
        
        for step in range(steps):
            # Random actions for testing (replace with trained policy later)
            action = self.environment.action_space.sample()
            obs, reward, terminated, truncated, info = self.environment.step(action)
            total_reward += reward
            
            if step % 100 == 0:
                self.logger.info(f"Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
            
            if terminated or truncated:
                self.logger.info(f"Episode ended at step {step}")
                obs, info = self.environment.reset()
                total_reward = 0
        
        self.logger.info(f"{Emojis.SUCCESS} Simulation completed")
        return total_reward
    
    def train_controller(self):
        """Step 7: Train Controller"""
        training = self.config.training
        if not training.training_enabled:
            self.logger.info("Training disabled - skipping controller training")
            return
        
        self.logger.info(f"{Emojis.BRAIN} Training controller with {training.algorithm}...")
        
        if not self.environment:
            raise ValueError("Simulation must be run before training controller")
        
        if training.training_backend == "stable_baselines3":
            self._train_with_sb3()
        elif training.training_backend == "rsl_rl":
            self._train_with_rsl_rl()
        else:
            raise ValueError(f"Unsupported training backend: {training.training_backend}")
        
        self.logger.info(f"{Emojis.SUCCESS} Controller training completed")
    
    def _train_with_sb3(self):
        """Train with Stable Baselines3."""
        training = self.config.training
        
        try:
            from stable_baselines3 import PPO, SAC, A2C
            from stable_baselines3.common.vec_env import DummyVecEnv
            
            # Wrap environment
            env = DummyVecEnv([lambda: self.environment])
            
            # Create model based on config
            if training.algorithm == "PPO":
                model = PPO(
                    "MlpPolicy", env,
                    learning_rate=training.learning_rate,
                    gamma=training.gamma,
                    gae_lambda=training.gae_lambda,
                    clip_range=training.clip_range,
                    ent_coef=training.ent_coef,
                    vf_coef=training.vf_coef,
                    batch_size=training.batch_size,
                    n_epochs=training.n_epochs,
                    verbose=1,
                    policy_kwargs={"net_arch": training.policy_network}
                )
            else:
                raise ValueError(f"Algorithm {training.algorithm} not supported yet")
            
            # Train
            model.learn(total_timesteps=training.total_timesteps)
            
            # Save model
            if self.config.output.save_trained_models:
                model_path = f"{self.config.output.log_dir}/trained_model"
                model.save(model_path)
                self.logger.info(f"Model saved to {model_path}")
            
            self.trained_model = model
            
        except ImportError:
            self.logger.error("Stable Baselines3 not available. Install with: pip install stable-baselines3")
            raise
    
    def _train_with_rsl_rl(self):
        """Train with RSL-RL (Genesis style)."""
        # TODO: Implement RSL-RL integration similar to Genesis examples
        raise NotImplementedError("RSL-RL integration not yet implemented")
    
    def export_firmware(self, export_path: str = None):
        """Step 8: Export Firmware for the Robot"""
        if not self.config.output.export_firmware:
            self.logger.info("Firmware export disabled - skipping")
            return
        
        self.logger.info(f"{Emojis.TOOLS} Exporting firmware...")
        
        if not self.trained_model:
            self.logger.warning("No trained model available - exporting base configuration only")
        
        export_path = export_path or f"{self.config.output.log_dir}/firmware"
        os.makedirs(export_path, exist_ok=True)
        
        # Export robot configuration
        robot_config_path = f"{export_path}/robot_config.json"
        with open(robot_config_path, 'w') as f:
            json.dump(asdict(self.robot_config), f, indent=2)
        
        # Export controller parameters
        controller_config = {
            "kp": self.robot_config.kp,
            "kd": self.robot_config.kd,
            "action_scale": self.robot_config.action_scale,
            "joint_names": self.robot_config.joint_names,
            "default_joint_angles": self.robot_config.default_joint_angles
        }
        
        controller_path = f"{export_path}/controller_config.json"
        with open(controller_path, 'w') as f:
            json.dump(controller_config, f, indent=2)
        
        # Export trained model if available
        if self.trained_model and hasattr(self.trained_model, 'save'):
            model_path = f"{export_path}/policy_model"
            self.trained_model.save(model_path)
        
        self.logger.info(f"{Emojis.SUCCESS} Firmware exported to {export_path}")
        return export_path
    
    def generate_simulation_report(self):
        """Step 9: Automated Report Generation"""
        if not self.config.output.generate_reports:
            self.logger.info("Report generation disabled - skipping")
            return
        
        self.logger.info(f"{Emojis.CHART} Generating simulation report...")
        
        report = {
            "configuration": self.config.to_dict(),
            "robot_config": asdict(self.robot_config) if self.robot_config else None,
            "reward_config": len(self.reward_manager.reward_functions) if self.reward_manager else 0,
            "training_completed": self.trained_model is not None,
            "environment_created": self.environment is not None,
        }
        
        # Add performance metrics if available
        if self.environment:
            report["environment_info"] = {
                "observation_space": str(self.environment.observation_space),
                "action_space": str(self.environment.action_space),
                "num_environments": self.config.task.num_parallel_envs
            }
        
        # Save report
        report_path = f"{self.config.output.log_dir}/simulation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"{Emojis.SUCCESS} Simulation report generated: {report_path}")
        return report
    
    def run_complete_pipeline(self, robot_config: Union[str, UniversalRobotConfig, dict] = None):
        """
        Run the complete simulation pipeline in sequence.
        
        This executes all steps from the workflow diagram:
        1-9 as defined in the individual methods.
        """
        self.logger.info(f"{Emojis.ROCKET} Starting complete simulation pipeline...")
        
        try:
            # Steps 1-2 are done in __init__
            
            # Step 3: Import Robot (use config if not provided)
            robot_cfg = robot_config or self.config.get_robot_config()
            self.import_robot(robot_cfg)
            
            # Step 4: Integrate Controller (use config values)
            controller_config = {
                "kp": self.config.robot.kp,
                "kd": self.config.robot.kd,
                "action_scale": self.config.robot.action_scale
            }
            self.integrate_controller(controller_config)
            
            # Step 5: Define Task and Rewards (use config)
            self.define_task_and_rewards()
            
            # Step 6: Run Simulation
            self.run_simulation()
            
            # Step 7: Train Controller
            self.train_controller()
            
            # Step 8: Export Firmware
            firmware_path = self.export_firmware()
            
            # Step 9: Generate Report
            report = self.generate_simulation_report()
            
            self.logger.info(f"{Emojis.SUCCESS} Complete simulation pipeline finished successfully!")
            
            return {
                "success": True,
                "firmware_path": firmware_path,
                "report": report,
                "model": self.trained_model
            }
            
        except Exception as e:
            self.logger.error(f"{Emojis.ERROR} Pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def cleanup(self, destroy_genesis=True):
        """Clean up resources."""
        if self.environment:
            self.environment.close()
        
        # Only destroy Genesis if explicitly requested (for final cleanup)
        if destroy_genesis and GENESIS_AVAILABLE and self.config.physics.physics_engine == "genesis":
            try:
                gs.destroy()
                self.logger.info("Genesis resources cleaned up")
            except:
                pass  # Genesis might not be initialized yet
        elif not destroy_genesis:
            self.logger.info("Environment closed, Genesis preserved for reuse")
                
        self.logger.info("Simulation stage cleaned up")


# Simple convenience function
def run_simulation_pipeline(config: UniversalPlatformConfig = None) -> dict:
    """
    Quick function to run complete simulation pipeline.
    
    Args:
        config: Universal platform configuration
    
    Returns:
        Dictionary with results including trained model and paths
    """
    
    # Use provided config or create default
    if config is None:
        from unified_platform.config.universal_config import PredefinedConfigs
        config = PredefinedConfigs.go2_locomotion()
    
    # Create and run simulation stage
    sim_stage = SimulationStage(config)
    
    try:
        result = sim_stage.run_complete_pipeline()
        return result
    finally:
        # Don't destroy Genesis here - just close environment
        # This allows Genesis to be reused by other tests
        sim_stage.cleanup(destroy_genesis=False)


if __name__ == "__main__":
    # Example usage
    print("🚀 Running Simulation Stage Example")
    
    # Quick test with Go2 robot using universal config
    from unified_platform.config.universal_config import PredefinedConfigs
    
    config = PredefinedConfigs.go2_locomotion()
    result = run_simulation_pipeline(config)
    
    if result["success"]:
        print("✅ Simulation pipeline completed successfully!")
        print(f"Firmware exported to: {result.get('firmware_path')}")
    else:
        print(f"❌ Pipeline failed: {result.get('error')}")
