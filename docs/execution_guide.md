# Execution Guide: Universal Robot Development Platform

This guide explains the available executable files, their purposes, and how to use them effectively.

## Quick Start

```powershell
# 1. Activate your environment
conda activate total_robotics  # if using conda
# OR
.\\venv\\Scripts\\activate     # if using venv

# 2. Run basic tests
python test_complete_flow.py
```

## Available Executables

### 1. Testing and Verification Files

#### `test_complete_flow.py`
Tests the complete pipeline of the platform.

```powershell
python test_complete_flow.py [--render] [--debug]
```
Options:
- `--render`: Enable visualization
- `--debug`: Show detailed debug information

Expected Output:
```
Testing environment creation... OK
Testing robot configuration... OK
Testing reward system... OK
...
All tests passed!
```

#### `run_test_sim_robot_arm.py`
Tests the robot arm simulation specifically.

```powershell
python run_test_sim_robot_arm.py [--render]
```
Options:
- `--render`: Enable visualization

### 2. Training Files

#### `training/professional_residual_training.py`
Main training script for the Go2 quadruped robot.

```powershell
python training/professional_residual_training.py [--render] [--episodes NUM] [--save-path PATH]
```
Options:
- `--render`: Enable visualization during training
- `--episodes`: Number of training episodes (default: 1000)
- `--save-path`: Path to save the trained model

Configuration:
- Uses PPO algorithm by default
- Creates checkpoints in `./checkpoints/`
- Logs to `./tensorboard_logs/`

#### `unified_platform/application/universal_train.py`
Universal training interface for any robot configuration.

```powershell
python unified_platform/application/universal_train.py --robot ROBOT_NAME --config CONFIG_PATH
```
Required Arguments:
- `--robot`: Name of the robot (e.g., "go2", "franka")
- `--config`: Path to robot configuration file

Optional Arguments:
- `--render`: Enable visualization
- `--episodes`: Number of training episodes
- `--algorithm`: RL algorithm to use (default: "PPO")

Example:
```powershell
python unified_platform/application/universal_train.py --robot go2 --config configs/go2_locomotion.json --render
```

#### `training/training_analysis.py`
Analyzes training results and generates performance reports.

```powershell
python training/training_analysis.py --log-dir PATH [--save-plots]
```
Options:
- `--log-dir`: Directory containing training logs
- `--save-plots`: Save analysis plots to disk

## Visualization and Monitoring

### TensorBoard
Monitor training progress in real-time:

```powershell
tensorboard --logdir tensorboard_logs
```
Then open http://localhost:6006 in your browser.

### Training Metrics
Available metrics include:
- Episode rewards
- Policy loss
- Value loss
- Learning rate
- Environment stats

## Common Workflows

### 1. Initial Setup Verification
```powershell
# 1. Verify basic functionality
python test_complete_flow.py

# 2. Test robot arm simulation
python run_test_sim_robot_arm.py
```

### 2. Training a Pre-configured Robot
```powershell
# Train Go2 quadruped
python training/professional_residual_training.py --render

# Monitor training
tensorboard --logdir tensorboard_logs
```

### 3. Training a Custom Robot
```powershell
# 1. Prepare configuration file
# configs/custom_robot.json

# 2. Start training
python unified_platform/application/universal_train.py --robot custom --config configs/custom_robot.json

# 3. Analyze results
python training/training_analysis.py --log-dir tensorboard_logs/custom_robot
```

## Troubleshooting

### Common Issues

1. **Visualization Problems**
   ```
   Solution: Check GPU drivers and try CPU-only mode
   ```

2. **Out of Memory**
   ```
   Solution: Reduce batch size in configuration
   ```

3. **Training Instability**
   ```
   Solution: Adjust learning rate and PPO parameters
   ```

### Debug Mode
Add `--debug` flag to most scripts for detailed output:
```powershell
python test_complete_flow.py --debug
```

## Best Practices

1. **Before Training**
   - Run all tests
   - Verify GPU availability
   - Check configuration files

2. **During Training**
   - Monitor TensorBoard
   - Save regular checkpoints
   - Track resource usage

3. **After Training**
   - Run analysis
   - Save model and configs
   - Document parameters

## Additional Resources

- Full API documentation in `docs/`
- CUDA setup guide in `docs/cuda_setup.md`
- Example configurations in `configs/`