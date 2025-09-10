"""
Universal Platform Configuration
===============================

Centralized configuration system that allows all parameters to be:
1. Set via class variables (for testing)
2. Loaded from JSON files
3. Passed from UI (future implementation)
4. Overridden at runtime

This eliminates all hardcoded values and makes the platform truly universal.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
import json
import os
from pathlib import Path


@dataclass
class UniversalRobotConfig:
    """Universal robot configuration that can come from UI, files, or code."""
    
    # Robot Identity
    name: str = "default_robot"
    robot_type: str = "builtin"  # "builtin", "urdf", "custom"
    robot_source: str = "go2"  # robot name or file path
    
    # Physical Properties
    urdf_path: Optional[str] = None
    joint_names: List[str] = field(default_factory=list)
    default_joint_angles: Dict[str, float] = field(default_factory=dict)
    num_actions: int = 12  # Number of controllable joints
    
    # Spawn Configuration
    base_init_pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.3])
    base_init_quat: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
    base_init_lin_vel: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    base_init_ang_vel: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    
    # Controller Parameters
    kp: float = 25.0
    kd: float = 1.0
    action_scale: float = 0.25
    clip_actions: float = 100.0
    torque_limit: float = 100.0
    
    # Episode Parameters  
    episode_length_s: float = 20.0
    dt: float = 0.02
    
    # Observation Configuration
    obs_scales: Dict[str, float] = field(default_factory=lambda: {
        "lin_vel": 2.0, "ang_vel": 0.25, 
        "dof_pos": 1.0, "dof_vel": 0.05
    })
    
    # Termination Conditions
    termination_conditions: Dict[str, Any] = field(default_factory=lambda: {
        "max_roll_degrees": 45.0,
        "max_pitch_degrees": 45.0,
        "joint_limit_margin": 0.1,
    })
    
    # Physics Properties
    mass_scale: float = 1.0
    friction_coefficient: float = 1.0
    restitution: float = 0.0


@dataclass
class UniversalTaskConfig:
    """Universal task configuration for any type of robotics task."""
    
    # Task Identity
    task_name: str = "default_task"
    task_type: str = "locomotion"  # "locomotion", "manipulation", "navigation", "custom"
    task_description: str = ""
    
    # Environment Configuration
    environment_type: str = "flat"  # "flat", "rough", "stairs", "obstacles", "maze"
    environment_size: List[float] = field(default_factory=lambda: [10.0, 10.0, 2.0])
    environment_params: Dict[str, Any] = field(default_factory=dict)
    
    # Episode Configuration
    max_episode_steps: int = 1000
    episode_timeout_seconds: float = 20.0
    num_parallel_envs: int = 1
    
    # Success Criteria
    success_conditions: Dict[str, Any] = field(default_factory=dict)
    termination_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UniversalRewardConfig:
    """Universal reward configuration system."""
    
    # Reward Components
    reward_components: List[Dict[str, Any]] = field(default_factory=list)
    
    # Reward Scaling
    total_reward_scale: float = 1.0
    reward_clip_min: float = -10.0
    reward_clip_max: float = 10.0
    
    # Default Locomotion Rewards
    use_default_locomotion: bool = True
    velocity_tracking_weight: float = 1.0
    action_smoothness_weight: float = -0.01
    joint_regularization_weight: float = -0.005
    collision_penalty_weight: float = -1.0
    
    # Default Manipulation Rewards
    use_default_manipulation: bool = False
    position_tracking_weight: float = 10.0
    orientation_tracking_weight: float = 5.0
    grasp_success_weight: float = 100.0


@dataclass
class UniversalTrainingConfig:
    """Universal training configuration for any RL algorithm."""
    
    # Training Control
    training_enabled: bool = True
    algorithm: str = "PPO"  # "PPO", "SAC", "TD3", "A2C"
    training_backend: str = "stable_baselines3"  # "stable_baselines3", "rsl_rl"
    
    # Training Duration
    total_timesteps: int = 10000
    max_training_hours: float = 1.0
    save_frequency: int = 1000
    
    # Algorithm Parameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # PPO Specific
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    n_epochs: int = 10
    
    # Network Architecture
    policy_network: List[int] = field(default_factory=lambda: [256, 256])
    value_network: List[int] = field(default_factory=lambda: [256, 256])
    activation_function: str = "tanh"


@dataclass
class UniversalPhysicsConfig:
    """Universal physics engine configuration."""
    
    # Physics Engine
    physics_engine: str = "genesis"  # "genesis", "isaac", "mujoco", "gazebo"
    backend: str = "gpu"  # "gpu", "cpu"
    
    # Simulation Parameters
    dt: float = 0.02  # Control frequency
    substeps: int = 2
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    
    # Solver Settings
    solver_iterations: int = 10
    contact_stiffness: float = 1e6
    contact_damping: float = 1e3


@dataclass
class UniversalRenderingConfig:
    """Universal rendering and visualization configuration."""
    
    # Rendering Control
    enable_rendering: bool = True
    render_mode: str = "human"  # "human", "rgb_array", "depth", "none"
    show_viewer: bool = True
    
    # Camera Settings
    camera_position: List[float] = field(default_factory=lambda: [2.0, 0.0, 2.5])
    camera_target: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.5])
    camera_fov: float = 40.0
    
    # Recording Settings
    record_video: bool = False
    video_fps: int = 30
    video_resolution: List[int] = field(default_factory=lambda: [1920, 1080])
    
    # Visual Effects
    show_grid: bool = True
    show_axes: bool = False
    lighting_quality: str = "medium"  # "low", "medium", "high"


@dataclass
class UniversalOutputConfig:
    """Universal output and export configuration."""
    
    # Logging
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    log_dir: str = "simulation_logs"
    enable_console_output: bool = True
    enable_file_output: bool = True
    
    # Model Saving
    save_trained_models: bool = True
    save_intermediate_models: bool = False
    model_save_format: str = "stable_baselines3"  # "stable_baselines3", "pytorch", "onnx"
    
    # Report Generation
    generate_reports: bool = True
    report_format: str = "json"  # "json", "html", "pdf"
    include_performance_metrics: bool = True
    include_training_curves: bool = True
    
    # Firmware Export
    export_firmware: bool = False
    firmware_format: str = "json"  # "json", "yaml", "binary"
    include_model_weights: bool = True


class UniversalPlatformConfig:
    """
    Main configuration class that combines all configuration aspects.
    This replaces all hardcoded values in the simulation system.
    """
    
    def __init__(self):
        """Initialize with default configurations."""
        self.robot = UniversalRobotConfig()
        self.task = UniversalTaskConfig()
        self.reward = UniversalRewardConfig()
        self.training = UniversalTrainingConfig()
        self.physics = UniversalPhysicsConfig()
        self.rendering = UniversalRenderingConfig()
        self.output = UniversalOutputConfig()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UniversalPlatformConfig':
        """Create configuration from dictionary (UI input)."""
        config = cls()
        
        if 'robot' in config_dict:
            config.robot = UniversalRobotConfig(**config_dict['robot'])
        if 'task' in config_dict:
            config.task = UniversalTaskConfig(**config_dict['task'])
        if 'reward' in config_dict:
            config.reward = UniversalRewardConfig(**config_dict['reward'])
        if 'training' in config_dict:
            config.training = UniversalTrainingConfig(**config_dict['training'])
        if 'physics' in config_dict:
            config.physics = UniversalPhysicsConfig(**config_dict['physics'])
        if 'rendering' in config_dict:
            config.rendering = UniversalRenderingConfig(**config_dict['rendering'])
        if 'output' in config_dict:
            config.output = UniversalOutputConfig(**config_dict['output'])
        
        return config
    
    @classmethod
    def from_json_file(cls, config_file: str) -> 'UniversalPlatformConfig':
        """Load configuration from JSON file."""
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for UI/JSON export."""
        return {
            'robot': asdict(self.robot),
            'task': asdict(self.task),
            'reward': asdict(self.reward),
            'training': asdict(self.training),
            'physics': asdict(self.physics),
            'rendering': asdict(self.rendering),
            'output': asdict(self.output)
        }
    
    def save_to_json(self, config_file: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def update_from_ui(self, ui_data: Dict[str, Any]):
        """Update configuration from UI input."""
        config_dict = self.to_dict()
        
        # Deep merge UI data into existing configuration
        for section, values in ui_data.items():
            if section in config_dict:
                config_dict[section].update(values)
        
        # Recreate configuration with updated values
        updated_config = self.from_dict(config_dict)
        
        # Update current instance
        self.robot = updated_config.robot
        self.task = updated_config.task
        self.reward = updated_config.reward
        self.training = updated_config.training
        self.physics = updated_config.physics
        self.rendering = updated_config.rendering
        self.output = updated_config.output
    
    def get_robot_config(self):
        """Get robot configuration as a compatible RobotConfig-like object."""
        if self.robot.robot_type == "builtin" and self.robot.robot_source in ROBOT_LIBRARY:
            # Use built-in robot configuration
            return ROBOT_LIBRARY[self.robot.robot_source]()
        else:
            # Return the current robot configuration as-is
            return self.robot
    
    def set_robot_from_builtin(self, robot_name: str):
        """Set robot configuration from built-in library."""
        if robot_name not in ROBOT_LIBRARY:
            available = ", ".join(ROBOT_LIBRARY.keys())
            raise ValueError(f"Unknown robot '{robot_name}'. Available: {available}")
        
        self.robot = ROBOT_LIBRARY[robot_name]()
    
    def set_robot_from_urdf(self, name: str, urdf_path: str, joint_names: List[str], 
                           default_joint_angles: Optional[Dict[str, float]] = None, **kwargs):
        """Set robot configuration from URDF parameters."""
        self.robot = RobotLibrary.custom(
            name=name,
            urdf_path=urdf_path, 
            joint_names=joint_names,
            default_joint_angles=default_joint_angles,
            **kwargs
        )
    
    def load_robot_from_urdf_file(self, urdf_path: str, robot_name: Optional[str] = None):
        """Auto-generate robot config from URDF file (requires Genesis)."""
        try:
            import genesis as gs
            
            # Create temporary scene to inspect URDF
            scene = gs.Scene(show_viewer=False)
            robot = scene.add_entity(gs.morphs.URDF(file=urdf_path))
            scene.build(n_envs=1)
            
            # Extract joint information
            joint_names = []
            for joint_name in robot.get_joint_names():
                if robot.get_joint(joint_name).type != "fixed":  # Skip fixed joints
                    joint_names.append(joint_name)
            
            robot_name = robot_name or urdf_path.split("/")[-1].split(".")[0]
            
            self.robot = RobotLibrary.custom(
                name=robot_name,
                urdf_path=urdf_path,
                joint_names=joint_names
            )
            
        except ImportError:
            raise ImportError("Genesis required for auto-config from URDF")
        except Exception as e:
            raise RuntimeError(f"Failed to load config from URDF: {e}")

# Built-in Robot Configurations
class RobotLibrary:
    """Library of predefined robot configurations."""
    
    @staticmethod
    def go2() -> UniversalRobotConfig:
        """Go2 quadruped robot configuration."""
        return UniversalRobotConfig(
            name="go2_quadruped",
            robot_type="builtin",
            robot_source="go2",
            urdf_path="urdf/go2/urdf/go2.urdf",
            joint_names=[
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            ],
            default_joint_angles={
                "FL_hip_joint": 0.0, "FR_hip_joint": 0.0, "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
                "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
                "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
            },
            num_actions=12,
            base_init_pos=[0.0, 0.0, 0.42],
            base_init_quat=[1.0, 0.0, 0.0, 0.0],
            kp=20.0,
            kd=0.5,
            action_scale=0.25,
            termination_conditions={
                "max_roll_degrees": 10.0,
                "max_pitch_degrees": 10.0,
            }
        )
    
    @staticmethod 
    def franka() -> UniversalRobotConfig:
        """Franka Panda robot arm configuration."""
        return UniversalRobotConfig(
            name="franka_panda",
            robot_type="builtin", 
            robot_source="franka",
            urdf_path="urdf/franka_panda/panda.urdf",
            joint_names=[
                "panda_joint1", "panda_joint2", "panda_joint3",
                "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
            ],
            default_joint_angles={
                "panda_joint1": 0.0, "panda_joint2": -0.785, "panda_joint3": 0.0,
                "panda_joint4": -2.356, "panda_joint5": 0.0, "panda_joint6": 1.571, "panda_joint7": 0.785
            },
            num_actions=7,
            base_init_pos=[0.0, 0.0, 0.0],  # Fixed base
            base_init_quat=[1.0, 0.0, 0.0, 0.0],
            kp=150.0,  # Stiffer control for arm
            kd=10.0,
            action_scale=0.1,  # Smaller movements for precision
            termination_conditions={
                "max_roll_degrees": 180.0,  # Arms can rotate more
                "max_pitch_degrees": 180.0,
                "joint_limit_margin": 0.05,
            }
        )
    
    @staticmethod
    def anymal() -> UniversalRobotConfig:
        """ANYmal quadruped robot configuration."""
        return UniversalRobotConfig(
            name="anymal",
            robot_type="builtin",
            robot_source="anymal", 
            urdf_path="urdf/anymal/anymal.urdf",
            joint_names=[
                "LF_HAA", "LF_HFE", "LF_KFE",
                "LH_HAA", "LH_HFE", "LH_KFE",
                "RF_HAA", "RF_HFE", "RF_KFE", 
                "RH_HAA", "RH_HFE", "RH_KFE",
            ],
            default_joint_angles={
                "LF_HAA": 0.0, "LF_HFE": 0.4, "LF_KFE": -0.8,
                "LH_HAA": 0.0, "LH_HFE": -0.4, "LH_KFE": 0.8,
                "RF_HAA": 0.0, "RF_HFE": 0.4, "RF_KFE": -0.8,
                "RH_HAA": 0.0, "RH_HFE": -0.4, "RH_KFE": 0.8,
            },
            num_actions=12,
            base_init_pos=[0.0, 0.0, 0.5],
            base_init_quat=[1.0, 0.0, 0.0, 0.0],
        )
    
    @staticmethod
    def custom(name: str, urdf_path: str, joint_names: List[str], 
               default_joint_angles: Optional[Dict[str, float]] = None,
               **kwargs) -> UniversalRobotConfig:
        """Create custom robot configuration."""
        if default_joint_angles is None:
            default_joint_angles = {joint: 0.0 for joint in joint_names}
        
        base_defaults = {
            "robot_type": "urdf",
            "robot_source": urdf_path,
            "num_actions": len(joint_names),
            "base_init_pos": [0.0, 0.0, 0.3],
            "base_init_quat": [1.0, 0.0, 0.0, 0.0],
            "kp": 20.0,
            "kd": 0.5,
            "action_scale": 0.25,
        }
        
        config_kwargs = {**base_defaults, **kwargs}
        
        return UniversalRobotConfig(
            name=name,
            urdf_path=urdf_path,
            joint_names=joint_names,
            default_joint_angles=default_joint_angles,
            **config_kwargs
        )

# Robot Registry
ROBOT_LIBRARY = {
    "go2": RobotLibrary.go2,
    "franka": RobotLibrary.franka,
    "anymal": RobotLibrary.anymal,
}

# Predefined configurations for common scenarios
class PredefinedConfigs:
    """Predefined configurations for common robotics scenarios."""
    
    @staticmethod
    def go2_locomotion() -> UniversalPlatformConfig:
        """Go2 quadruped locomotion configuration."""
        config = UniversalPlatformConfig()
        
        # Use built-in robot configuration
        config.robot = RobotLibrary.go2()
        
        # Task configuration
        config.task.task_name = "Go2 Locomotion"
        config.task.task_type = "locomotion"
        config.task.environment_type = "flat"
        config.task.max_episode_steps = 1000
        
        # Training configuration
        config.training.total_timesteps = 50000
        config.training.algorithm = "PPO"
        
        return config
    
    @staticmethod
    def franka_manipulation() -> UniversalPlatformConfig:
        """Franka Panda manipulation configuration."""
        config = UniversalPlatformConfig()
        
        # Use built-in robot configuration
        config.robot = RobotLibrary.franka()
        
        # Task configuration
        config.task.task_name = "Franka Manipulation"
        config.task.task_type = "manipulation"
        config.task.environment_type = "table"
        config.task.max_episode_steps = 500
        
        # Reward configuration
        config.reward.use_default_locomotion = False
        config.reward.use_default_manipulation = True
        
        return config
    
    @staticmethod
    def anymal_locomotion() -> UniversalPlatformConfig:
        """ANYmal quadruped locomotion configuration."""
        config = UniversalPlatformConfig()
        
        # Use built-in robot configuration
        config.robot = RobotLibrary.anymal()
        
        # Task configuration
        config.task.task_name = "ANYmal Locomotion"
        config.task.task_type = "locomotion"
        config.task.environment_type = "rough"
        config.task.max_episode_steps = 1000
        
        return config
    
    @staticmethod
    def custom_robot_template() -> UniversalPlatformConfig:
        """Template for custom robot configuration."""
        config = UniversalPlatformConfig()
        
        # Robot configuration - to be filled by user/UI
        config.robot.name = "custom_robot"
        config.robot.robot_type = "urdf"
        config.robot.robot_source = "path/to/robot.urdf"
        
        # Task configuration - to be customized
        config.task.task_name = "Custom Task"
        config.task.task_type = "custom"
        
        return config


if __name__ == "__main__":
    # Example usage
    print("🔧 Universal Platform Configuration System")
    print("=" * 50)
    
    # Create configuration
    config = PredefinedConfigs.go2_locomotion()
    
    print("📋 Configuration created:")
    print(f"Robot: {config.robot.name}")
    print(f"Task: {config.task.task_name}")
    print(f"Training: {config.training.algorithm} for {config.training.total_timesteps} steps")
    
    # Save to file
    config.save_to_json("configs/go2_locomotion.json")
    print("💾 Configuration saved to file")
    
    # Load from file
    loaded_config = UniversalPlatformConfig.from_json_file("configs/go2_locomotion.json")
    print("📂 Configuration loaded from file")
    
    # Simulate UI update
    ui_update = {
        "training": {
            "total_timesteps": 100000,
            "learning_rate": 1e-4
        },
        "robot": {
            "kp": 30.0
        }
    }
    
    config.update_from_ui(ui_update)
    print("🌐 Configuration updated from UI")
    print(f"New timesteps: {config.training.total_timesteps}")
    print(f"New kp: {config.robot.kp}")


# Utility Functions for Easy Configuration Creation
def make_robot_config(robot_name: str) -> UniversalRobotConfig:
    """Get built-in robot configuration by name."""
    if robot_name not in ROBOT_LIBRARY:
        available = ", ".join(ROBOT_LIBRARY.keys())
        raise ValueError(f"Unknown robot '{robot_name}'. Available: {available}")
    
    return ROBOT_LIBRARY[robot_name]()


def make_custom_robot_config(name: str, urdf_path: str, joint_names: List[str],
                           default_joint_angles: Optional[Dict[str, float]] = None,
                           **kwargs) -> UniversalRobotConfig:
    """Create custom robot configuration."""
    return RobotLibrary.custom(
        name=name,
        urdf_path=urdf_path,
        joint_names=joint_names,
        default_joint_angles=default_joint_angles,
        **kwargs
    )


def make_universal_config(
    robot_name_or_config: Union[str, UniversalRobotConfig],
    task_type: str = "locomotion",
    training_timesteps: int = 50000,
    **kwargs
) -> UniversalPlatformConfig:
    """Quickly create a universal configuration."""
    config = UniversalPlatformConfig()
    
    # Set robot configuration
    if isinstance(robot_name_or_config, str):
        config.robot = make_robot_config(robot_name_or_config)
    else:
        config.robot = robot_name_or_config
    
    # Set task configuration
    config.task.task_type = task_type
    config.task.task_name = f"{config.robot.name.title()} {task_type.title()}"
    
    # Set training configuration
    config.training.total_timesteps = training_timesteps
    
    # Apply any additional parameters
    for section, values in kwargs.items():
        if hasattr(config, section):
            section_obj = getattr(config, section)
            for key, value in values.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
    
    return config
