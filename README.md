# Universal Robot Development Platform

A flexible and universal platform for robot development, simulation, and training using Genesis physics engine and Gymnasium environments.

## Table of Contents

- [Overview](#overview)
- [Core Architecture](#core-architecture)
- [Setup Guide](#setup-guide)
- [Usage Guide](#usage-guide)
- [Custom Robot Integration](#custom-robot-integration)
- [Best Practices](#best-practices)

## Overview

The Universal Robot Development Platform is designed to:

1. **Make Robot Training Easy**:

   - Converts robot designs (URDF files) into training environments
   - Provides standardized Gymnasium interface for AI training
   - Simple one-line setup for common robots

2. **Support Any Robot**:

   - Configuration-driven approach, no hardcoding
   - Works with any URDF-described robot
   - Flexible reward system adaptation

3. **Optimize for AI Training**:
   - Genesis physics for realistic simulation
   - GPU-accelerated computations
   - Integration with popular ML frameworks

## Core Architecture

### Main Components

1. **Root Level** (`/total_robotics/`)

   - High-level test files and examples
   - Main entry points for users
   - Example: `test_complete_flow.py`

2. **Unified Platform** (`/unified_platform/`)
   - `application/`: Training and simulation apps
   - `config/`: Configuration management
   - `environment/`: Robot environment implementations
3. **Robot Descriptions** (`/config/go2_description/`, etc.)

   - URDF files and robot configurations
   - Mesh files and physical descriptions
   - Launch configurations

4. **Training** (`/training/`)
   - Training scripts and analysis tools
   - Professional residual training
   - Training metrics and analysis

## Setup Guide

### Prerequisites

#### Required Software

- Python 3.10 or higher
- Git
- NVIDIA GPU with CUDA support (recommended) or CPU-only setup

#### Hardware Requirements

- Minimum 8GB RAM (16GB recommended)
- NVIDIA GPU with 6GB VRAM for training (recommended)
- CPU: 4+ cores recommended

### Environment Setup

We support both `conda` and `venv` for environment management. Choose the method that works best for you.

#### Option 1: Using Conda (Recommended)

```powershell
# 1. Clone repository
git clone https://github.com/Total-Bots-Lab/total_robotics.git
cd total_robotics

# 2. Create conda environment
conda create -n total_robotics python=3.10
conda activate total_robotics

# 3. Install PyTorch with CUDA (for GPU support)
# For CUDA 11.8 (default)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# OR for CPU only
# conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 4. Install other dependencies
pip install -r requirements.txt
```

#### Option 2: Using venv

```powershell
# 1. Clone repository
git clone https://github.com/Total-Bots-Lab/total_robotics.git
cd total_robotics

# 2. Create and activate virtual environment
# Windows:
python -m venv venv
.\\venv\\Scripts\\activate

# Linux/Mac:
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Dependency Management

The project uses several key components:

1. **Core Dependencies**:

   - Genesis Physics Engine (v0.3.1)
   - PyTorch (with CUDA support)
   - Gymnasium for RL environments
   - Stable-Baselines3 for RL algorithms

2. **Visualization Tools**:

   - TensorBoard for training visualization
   - PyVista for 3D visualization
   - OpenCV for image processing

3. **Robot Simulation**:
   - MuJoCo physics engine
   - URDF parser for robot descriptions
   - Trimesh for mesh processing

### CUDA Setup

For GPU acceleration (recommended for training):

1. **Check GPU and CUDA Compatibility**:

   ```powershell
   nvidia-smi
   ```

2. **Follow CUDA Setup Guide**:

   - Detailed instructions in [docs/cuda_setup.md](docs/cuda_setup.md)
   - Includes GPU compatibility table
   - Troubleshooting common issues

3. **Verify CUDA Installation**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   ```

### Genesis Physics Engine Setup

```powershell
# 1. Install Genesis
pip install genesis-world==0.3.1

# 2. Verify installation
python -c "import genesis as gs; print(gs.__version__)"

# 3. Configure for CPU-only (if needed)
# Windows:
$env:GENESIS_CPU_ONLY = "1"
# Linux/Mac:
export GENESIS_CPU_ONLY=1
```

### Verifying Installation

Run the test suite to verify your setup:

```powershell
# Run basic tests
pytest test_complete_flow.py

# Run robot arm simulation test
python run_test_sim_robot_arm.py
```

Expected output:

```
All tests passed! Environment is properly set up.
```

### Troubleshooting Common Issues

1. **CUDA/PyTorch Issues**:

   - Ensure NVIDIA drivers are up to date
   - Check CUDA version compatibility
   - See [docs/cuda_setup.md](docs/cuda_setup.md)

2. **Import Errors**:

   - Verify virtual environment is activated
   - Check Python version: `python --version`
   - Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

3. **Genesis Engine Issues**:
   - Try CPU-only mode first
   - Check GPU memory usage
   - Verify CUDA toolkit installation

### Getting Help

If you encounter any issues:

1. Check the documentation in the `docs/` folder
2. Run verification tests
3. Check CUDA setup guide
4. Open an issue on GitHub with:
   - Your environment details
   - Error messages
   - Steps to reproduce

## Usage Guide

### Basic Usage

1. **Run Tests**:

   ```powershell
   python test_complete_flow.py
   python run_test_sim_robot_arm.py
   ```

2. **Train a Pre-defined Robot**:

   ```python
   from unified_platform.environment import make_robot_env
   from unified_platform.config import RobotLibrary
   from stable_baselines3 import PPO

   # Create environment
   robot_config = RobotLibrary.go2()  # For Go2 quadruped
   env = make_robot_env("go2", render_mode="human")

   # Train model
   model = PPO("MlpPolicy", env)
   model.learn(total_timesteps=1000000)

   # Save model
   model.save("my_robot_model")
   ```

3. **Visualize Training**:
   ```powershell
   tensorboard --logdir tensorboard_logs
   ```

## Custom Robot Integration

### Adding New Robots

1. **Prepare Robot Description**:

   - Add URDF to `config/your_robot_description/urdf/`
   - Add meshes to `config/your_robot_description/meshes/`

2. **Create Configuration**:

   ```json
   {
     "joint_names": ["joint1", "joint2"],
     "joint_pos_limits": [[-3.14, 3.14]],
     "kp": 40.0,
     "kd": 3.0,
     "action_scale": 1.0
   }
   ```

3. **Use Custom Robot**:

   ```python
   from unified_platform.environment import make_custom_env

   env = make_custom_env(
       urdf_path="path/to/your/robot.urdf",
       joint_names=["joint1", "joint2"],
       config_path="configs/your_robot.json"
   )
   ```

### Custom Reward Functions

1. Create reward class:

   ```python
   class CustomReward(GenericQuadrupedReward):
       def compute(self, env_state):
           # Implement custom reward logic
           return reward_value
   ```

2. Register with RewardManager:
   ```python
   reward_manager = RewardManager([CustomReward()])
   env = make_robot_env("go2", reward_manager=reward_manager)
   ```

## Best Practices

### Performance Optimization

- Use GPU acceleration when available
- Adjust batch sizes based on memory
- Monitor with TensorBoard

### Resource Management

- Always use virtual environment
- Regular model checkpointing
- Clean up Genesis resources:
  ```python
  env.close()
  ```

### Training Tips

- Start with pre-defined robots
- Use incremental reward functions
- Monitor training metrics
- Save checkpoints frequently

## Troubleshooting

Common issues and solutions:

1. **CUDA Issues**:

   - Verify CUDA installation
   - Check PyTorch CUDA compatibility
   - Try CPU-only mode for testing

2. **Environment Issues**:

   - Verify Python version
   - Check virtual environment activation
   - Validate all dependencies installed

3. **Training Issues**:
   - Monitor resource usage
   - Check reward scaling
   - Verify environment reset behavior

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Submit pull request

## License

[Add appropriate license information]

## Contact

Total-Robotics Team
