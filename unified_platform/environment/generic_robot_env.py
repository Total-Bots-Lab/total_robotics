"""
Generic Genesis Gym Environment
===============================

Universal Gymnasium environment that works with any robot URDF file.
Users only need to provide robot configuration and reward setup.
"""

import gymnasium as gym
import numpy as np
import torch
import math
from typing import Optional, Tuple, Dict, Any

# Genesis imports
try:
    import genesis as gs
    from genesis.utils.geom import inv_quat, transform_by_quat, quat_to_xyz, transform_quat_by_quat
    GENESIS_AVAILABLE = True
except ImportError:
    GENESIS_AVAILABLE = False
    print("Warning: Genesis not available. Environment will not function properly.")

from unified_platform.config.universal_config import UniversalRobotConfig, make_robot_config, make_universal_config
from unified_platform.config.reward_system import RewardManager, create_locomotion_rewards, create_manipulation_rewards


def gs_rand_float(lower, upper, shape, device):
    """Random float generation helper function."""
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class GenericRobotGymEnv(gym.Env):
    """
    Universal Genesis Robot Environment.
    
    Works with any robot URDF file. Users provide:
    1. Robot configuration (joints, URDF path, etc.)
    2. Reward system (optional, defaults provided)
    3. Observation configuration (optional, auto-generated)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self, 
        robot_config: UniversalRobotConfig,
        reward_manager: Optional[RewardManager] = None,
        num_envs: int = 1, 
        render_mode: Optional[str] = "human",
        obs_components: Optional[list] = None,
        show_viewer: bool = True,
        stand_warmup_steps: int = 50,
    ):
        """Initialize generic robot environment."""
        super().__init__()
        
        if not GENESIS_AVAILABLE:
            raise ImportError("Genesis is required but not available.")
        
        self.robot_config = robot_config
        self.num_envs = num_envs
        self.render_mode = render_mode
        self.show_viewer = show_viewer
        self.stand_warmup_steps = stand_warmup_steps
        
        # Initialize Genesis first
        try:
            # Check if Genesis is already initialized
            gs.init(backend=gs.gpu)
        except Exception as e:
            if "already initialized" in str(e):
                pass  # Genesis already initialized
            else:
                raise e
        
        self.device = gs.device
        
        # Environment parameters from config
        self.dt = robot_config.dt
        self.max_episode_length = math.ceil(robot_config.episode_length_s / self.dt)
        self.num_actions = robot_config.num_actions
        
        # Set up reward system
        if reward_manager is None:
            # Auto-detect robot type and create appropriate rewards
            if "arm" in robot_config.name.lower() or "franka" in robot_config.name.lower():
                self.reward_manager = create_manipulation_rewards(self.dt)
            else:
                self.reward_manager = create_locomotion_rewards(self.dt)
        else:
            self.reward_manager = reward_manager
        
        # Set up observation components
        if obs_components is None:
            self.obs_components = self._get_default_obs_components()
        else:
            self.obs_components = obs_components
        
        # Calculate observation size
        self.num_obs = self._calculate_obs_size()
        
        # Define Gymnasium spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.num_actions,), 
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.num_obs,), 
            dtype=np.float32
        )
        
        # Initialize Genesis scene and robot
        self._setup_scene()
        self._setup_robot()
        self._setup_buffers()
        
    def _get_default_obs_components(self) -> list:
        """Get default observation components based on robot type."""
        if "arm" in self.robot_config.name.lower() or "franka" in self.robot_config.name.lower():
            # Manipulation robot observations
            return [
                {"name": "joint_pos", "size": self.num_actions, "scale": 1.0},
                {"name": "joint_vel", "size": self.num_actions, "scale": 0.1},
                {"name": "actions", "size": self.num_actions, "scale": 1.0},
            ]
        else:
            # Locomotion robot observations  
            return [
                {"name": "base_ang_vel", "size": 3, "scale": 0.25},
                {"name": "projected_gravity", "size": 3, "scale": 1.0},
                {"name": "joint_pos", "size": self.num_actions, "scale": 1.0},
                {"name": "joint_vel", "size": self.num_actions, "scale": 0.05},
                {"name": "actions", "size": self.num_actions, "scale": 1.0},
            ]
    
    def _calculate_obs_size(self) -> int:
        """Calculate total observation size."""
        return sum(comp["size"] for comp in self.obs_components)
    
    def _setup_scene(self):
        """Initialize Genesis physics scene with viewer enabled by default."""
        # Simple viewer setup - always show viewer for easy debugging
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            show_viewer=self.show_viewer,  # Use parameter instead of hardcoded True
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
        )
        
        # Add ground (generic)
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        
    def _setup_robot(self):
        """Setup robot from configuration."""
        # Robot initialization
        self.base_init_pos = torch.tensor(self.robot_config.base_init_pos, device=self.device)
        self.base_init_quat = torch.tensor(self.robot_config.base_init_quat, device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        
        # Load robot from URDF or XML
        robot_file_path = self.robot_config.urdf_path
        
        # Determine the appropriate Genesis morph based on file extension
        if robot_file_path.endswith('.xml'):
            # Use MJCF for XML files (MuJoCo format)
            robot_morph = gs.morphs.MJCF(
                file=robot_file_path,
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            )
        elif robot_file_path.endswith('.urdf'):
            # Use URDF for URDF files
            robot_morph = gs.morphs.URDF(
                file=robot_file_path,
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            )
        else:
            raise ValueError(f"Unsupported robot file format: {robot_file_path}")
        
        self.robot = self.scene.add_entity(robot_morph)
        
        # Build scene
        self.scene.build(n_envs=self.num_envs)
        
        # Motor setup
        self.motors_dof_idx = [
            self.robot.get_joint(name).dof_start 
            for name in self.robot_config.joint_names
        ]
        self.default_dof_pos = torch.tensor(
            [self.robot_config.default_joint_angles[name] for name in self.robot_config.joint_names],
            device=self.device, dtype=gs.tc_float
        )
        
        # PD control
        self.robot.set_dofs_kp([self.robot_config.kp] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.robot_config.kd] * self.num_actions, self.motors_dof_idx)
        
    def _setup_buffers(self):
        """Initialize state buffers."""
        # Episode tracking
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.int)
        
        # Robot state buffers
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        
        # Joint state buffers
        self.dof_pos = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        
        # Observation buffer
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        
        # Initialize reward manager
        self.reward_manager.reset_episode_sums(self.num_envs, self.device)
        
        # Extras for compatibility
        self.extras = {"observations": {}, "time_outs": torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)}
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)
        
        # Reset all environments
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
        # Get observation and info
        observation = self._get_observation()
        info = {"episode_length": 0}
        
        # Return single env format if needed
        if self.num_envs == 1:
            observation = observation[0].cpu().numpy()
        else:
            observation = observation.cpu().numpy()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one timestep."""
        # Convert action to tensor
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.device)
        
        if self.num_envs == 1 and len(action.shape) == 1:
            action = action.unsqueeze(0)

        # Apply action (residual around default) - FOLLOWING ARGO-ROBOT IMPLEMENTATION
        self.actions = torch.clip(action, -self.robot_config.clip_actions, self.robot_config.clip_actions)
        
        # EXACT ARGO-ROBOT FORMULA: target = action * scale + default
        # With optional action latency simulation (they use last_actions on real robot)
        exec_actions = self.last_actions if hasattr(self, 'simulate_action_latency') and self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.robot_config.action_scale + self.default_dof_pos
        
        # ADDITIONAL HIP JOINT LIMITING: Apply custom joint position limits if configured
        if hasattr(self.robot_config, 'joint_pos_limits') and self.robot_config.joint_pos_limits:
            for i, joint_name in enumerate(self.robot_config.joint_names):
                if joint_name in self.robot_config.joint_pos_limits:
                    min_pos, max_pos = self.robot_config.joint_pos_limits[joint_name]
                    target_dof_pos[:, i] = torch.clamp(target_dof_pos[:, i], min_pos, max_pos)
        
        # NO MANUAL JOINT CLAMPING - Genesis handles joint limits automatically
        # This was causing the stiffness - let Genesis handle the limits naturally
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()
        
        # Update state
        self._update_state()
        self.episode_length_buf += 1
        
        # Check termination
        terminated, truncated = self._check_termination()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Handle resets
        reset_envs = (terminated | truncated).nonzero(as_tuple=False).flatten()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)
        
        # Get observation
        observation = self._get_observation()
        info = {"episode_length": self.episode_length_buf[0].item() if self.num_envs == 1 else self.episode_length_buf}
        
        # Store previous actions
        self.last_actions = self.actions.clone()
        
        # Convert to Gymnasium format
        if self.num_envs == 1:
            observation = observation[0].cpu().numpy()
            reward = reward[0].item()
            terminated = terminated[0].item()
            truncated = truncated[0].item()
        else:
            observation = observation.cpu().numpy()
            reward = reward.cpu().numpy()
            terminated = terminated.cpu().numpy()
            truncated = truncated.cpu().numpy()
        
        return observation, reward, terminated, truncated, info
    
    def _update_state(self):
        """Update robot state buffers."""
        # Base state
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        
        # Compute orientation
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True, degrees=True,
        )
        
        # Transform velocities to robot frame
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        
        # Joint state
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        
    def _get_observation(self) -> torch.Tensor:
        """Generate observation vector based on configuration."""
        obs_parts = []
        
        for comp in self.obs_components:
            if comp["name"] == "base_ang_vel":
                obs_parts.append(self.base_ang_vel * comp["scale"])
            elif comp["name"] == "projected_gravity":
                obs_parts.append(self.projected_gravity * comp["scale"])
            elif comp["name"] == "joint_pos":
                obs_parts.append((self.dof_pos - self.default_dof_pos) * comp["scale"])
            elif comp["name"] == "joint_vel":
                obs_parts.append(self.dof_vel * comp["scale"])
            elif comp["name"] == "actions":
                obs_parts.append(self.actions * comp["scale"])
            # Add more observation components as needed
        
        self.obs_buf = torch.cat(obs_parts, dim=-1)
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf
    
    def _compute_reward(self) -> torch.Tensor:
        """Compute reward using reward manager."""
        # Prepare environment state for reward computation
        env_state = {
            "batch_size": self.num_envs,
            "device": self.device,
            "base_pos": self.base_pos,
            "base_quat": self.base_quat,
            "base_euler": self.base_euler,
            "base_lin_vel": self.base_lin_vel,
            "base_ang_vel": self.base_ang_vel,
            "dof_pos": self.dof_pos,
            "dof_vel": self.dof_vel,
            "default_dof_pos": self.default_dof_pos,
            "actions": self.actions,
            "last_actions": self.last_actions,
        }
        
        return self.reward_manager.compute_total_reward(env_state)
    
    def _check_termination(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions based on robot configuration."""
        # Time limit
        truncated = self.episode_length_buf > self.max_episode_length
        
        # Robot-specific termination conditions
        terminated = torch.zeros_like(truncated)
        
        if "max_roll_degrees" in self.robot_config.termination_conditions:
            max_roll = self.robot_config.termination_conditions["max_roll_degrees"]
            terminated |= torch.abs(self.base_euler[:, 0]) > max_roll
        
        if "max_pitch_degrees" in self.robot_config.termination_conditions:
            max_pitch = self.robot_config.termination_conditions["max_pitch_degrees"]
            terminated |= torch.abs(self.base_euler[:, 1]) > max_pitch
        
        # Handle timeouts
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"][:] = 0.0
        self.extras["time_outs"][time_out_idx] = 1.0
        
        return terminated, truncated
    
    def reset_idx(self, envs_idx):
        """Reset specific environments."""
        if len(envs_idx) == 0:
            return
        
        # Reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        
        # Reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)
        
        # Reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        
        # Reset reward tracking
        for name in self.reward_manager.episode_sums:
            self.reward_manager.episode_sums[name][envs_idx] = 0.0
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            pass  # Genesis handles this automatically
        elif self.render_mode == "rgb_array":
            pass  # Could implement camera capture here
        return None
    
    def close(self):
        """Close the environment."""
        if hasattr(self, 'scene'):
            try:
                # Properly close the Genesis scene
                if self.scene is not None:
                    self.scene.reset()  # Reset scene state
                    self.scene = None
                    print("🧹 Genesis scene closed")
            except Exception as e:
                print(f"⚠️  Scene cleanup warning: {e}")
        
        # Clear any cached data
        if hasattr(self, 'robot'):
            self.robot = None
        if hasattr(self, '_obs_buffer'):
            self._obs_buffer = None


# Convenience functions for easy usage
def make_robot_env(robot_name: str, **kwargs):
    """Create environment for a predefined robot."""
    robot_config = make_robot_config(robot_name)
    return GenericRobotGymEnv(robot_config, **kwargs)


def make_custom_env(urdf_path: str, joint_names: list, **kwargs):
    """Create environment for a custom robot."""
    from unified_platform.config.universal_config import RobotLibrary
    robot_config = RobotLibrary.custom(
        name=urdf_path.split("/")[-1].split(".")[0],
        urdf_path=urdf_path,
        joint_names=joint_names,
        **kwargs
    )
    return GenericRobotGymEnv(robot_config, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("Testing Go2 robot...")
    env = make_robot_env("go2", render_mode="human")
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    for i in range(10):
        action = np.random.uniform(-1, 1, size=env.num_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}")
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("Test completed!")
