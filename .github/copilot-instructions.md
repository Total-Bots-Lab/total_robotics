# Copilot Instructions: Unified Robotics Platform

## Architecture Overview
This is a **Universal Robot Development Platform** that converts Genesis physics simulations into standard Gymnasium environments for reinforcement learning training. The platform implements a complete 9-step robotics simulation pipeline with robot-agnostic design.

### Core Design Principles
- **Universal**: Works with ANY robot URDF file through configuration, not code changes
- **No Hardcoded Values**: All parameters configurable via JSON/UI (see `UniversalPlatformConfig`)
- **Direct Training Path**: Most users bypass pipeline complexity and go straight to training via `make_robot_env()`

## Key Architecture Layers

### 1. Configuration Layer (`unified_platform/config/`)
- **`universal_config.py`**: Central configuration system with `RobotLibrary` for predefined robots
- **Pattern**: Use `RobotLibrary.go2()` for built-in robots, never hardcode robot parameters
- **UI Ready**: All configs support JSON serialization via `to_dict()`/`from_dict()`

### 2. Environment Layer (`unified_platform/environment/`)
- **`generic_robot_env.py`**: Universal Gymnasium interface wrapping Genesis physics
- **Entry Points**: 
  - `make_robot_env("go2")` - predefined robots
  - `make_custom_env(urdf_path, joint_names, ...)` - custom URDFs
- **Genesis Integration**: Handles GPU/CPU backend, scene management, proper cleanup with `gs.destroy()`

### 3. Reward System Pattern
- **Professional**: Use `GenericQuadrupedReward` (modular, GPU-optimized, configurable)
- **Adapter Pattern**: Wrap external reward systems with `compute(env_state)` interface for `RewardManager`
- **Configuration**: Reward scales and components defined in `RewardConfig` dataclass

## Critical Workflows

### Training Script Structure
```python
# Standard pattern in examples/complete_training_integration.py
robot_config = RobotLibrary.go2()  # Never hardcode robot params
reward_manager = RewardManager()
env = make_robot_env("go2", render_mode="human", reward_manager=reward_manager)
model = PPO("MlpPolicy", env)  # Standard SB3 integration
```

### Genesis Environment Management
- **Single Environment Pattern**: Create one environment, reuse across episodes to avoid scene conflicts
- **Cleanup**: Always call `env.close()` and `gs.destroy()` in finally blocks
- **Device Handling**: Genesis tensors require proper GPU/CPU device management

### Configuration-Driven Development
- **Robot Addition**: Add to `RobotLibrary` class, never modify core environment code
- **Parameter Tuning**: Use `robot_config.kp`, `robot_config.kd`, `robot_config.action_scale` 
- **UI Integration**: All configs auto-serialize to JSON for frontend consumption

## Testing & Validation

### Test Files Pattern
- **`test_complete_flow.py`**: Integration test for entire pipeline
- **`examples/complete_training_integration.py`**: Production training workflow
- **Debug Pattern**: Use Unicode-safe logging via `setup_unicode_logger()` for emoji-rich output

### Common Issues to Avoid
1. **Genesis Scene Conflicts**: Don't create multiple Genesis scenes, use single environment with resets
2. **Tensor Device Mismatch**: Ensure consistent CUDA/CPU device handling in reward systems
3. **Joint Limits**: Always set `joint_pos_limits` for stability, especially hip joints on quadrupeds
4. **Import Path**: Use `sys.path.append()` pattern for unified_platform imports

## Integration Points

### External Dependencies
- **Genesis Physics**: Core simulator, version 0.3.1+
- **Stable Baselines3**: Standard RL library integration via Gymnasium interface
- **PyTorch**: For reward computations and tensor operations

### Robot-Specific Patterns
- **Go2 Quadruped**: Use symmetric joint limits, natural standing pose in `default_joint_pos`
- **Locomotion Training**: Reward forward velocity, height maintenance, natural gait patterns
- **Control Gains**: Start with `kp=40.0, kd=3.0` for quadrupeds, tune based on behavior

## Current Implementation Status
- ✅ **Simulation Pipeline**: Complete 9-step workflow implemented
- ✅ **Gymnasium Integration**: Full RL library compatibility  
- ✅ **Multi-Robot Support**: Go2, Franka, custom URDF support
- 🔄 **Production Use**: Direct training path bypasses pipeline complexity for most users
- ⏳ **Future**: Mechanical design, hardware deployment stages planned but not implemented

When working with this codebase, prioritize the direct training approach (`make_robot_env` → PPO training) unless specifically working on pipeline orchestration features.