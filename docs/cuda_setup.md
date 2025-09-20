# CUDA Setup Guide

## Checking Your GPU and Compatible CUDA Version

1. **Check Your NVIDIA GPU**:
   ```powershell
   nvidia-smi
   ```
   This will show your GPU model and the CUDA version currently installed.

2. **Find Compatible CUDA Version**:
   - Visit [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
   - Match your GPU with compatible CUDA versions
   - Current project uses CUDA 11.8, but can be adjusted based on your GPU

## Installing PyTorch with Specific CUDA Version

### Method 1: Using pip
```powershell
# For CUDA 11.8 (current project default)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip3 install torch torchvision torchaudio
```

### Method 2: Using conda
```powershell
# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# For CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## Verifying CUDA Setup

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Common CUDA Versions and GPU Compatibility

- CUDA 12.x: RTX 40-series, RTX 30-series, RTX 20-series
- CUDA 11.x: RTX 30-series, RTX 20-series, GTX 16-series, GTX 10-series
- CUDA 10.x: GTX 16-series, GTX 10-series, GTX 900-series

## Troubleshooting

1. **CUDA Version Mismatch**:
   - Uninstall existing PyTorch: `pip uninstall torch torchvision torchaudio`
   - Install the correct version using commands above

2. **Multiple CUDA Versions**:
   - Check system PATH for multiple CUDA installations
   - Set CUDA_HOME environment variable to desired version:
     ```powershell
     # PowerShell
     $env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
     $env:PATH = "$env:CUDA_HOME\bin;" + $env:PATH
     ```

3. **Genesis Physics Engine Compatibility**:
   - Genesis 0.3.1 works best with CUDA 11.x
   - For CPU-only: Set environment variable
     ```powershell
     $env:GENESIS_CPU_ONLY = "1"
     ```

## Project-Specific Notes

Our project uses CUDA 11.8 by default because:
- Stable compatibility with Genesis Physics Engine
- Wide GPU support range
- Proven stability with PyTorch 2.5.1

If you need to use a different CUDA version:

1. Update `requirements.txt`:
   ```txt
   # Replace these versions as needed
   torch==2.5.1+cu118  # Change cu118 to your CUDA version
   torchvision==0.20.1+cu118
   torchaudio==2.5.1+cu118
   ```

2. Install specific CUDA toolkit from NVIDIA website

3. Test compatibility:
   ```python
   from unified_platform.environment import make_robot_env
   env = make_robot_env("go2", render_mode="human")
   print("Setup successful!")
   ```