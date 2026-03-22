# Subconscious Robotics

Production-ready Sim-to-Real framework for robot training using PyBullet simulation with domain randomization for sim-to-real transfer.

## Features

- 🧠 **Subconscious Training**: Train robot policies using PPO/SAC with parallelized environments
- 🎲 **Domain Randomization**: Automatic physics parameter randomization (mass, friction, gravity) for sim-to-real transfer
- 🎯 **Multi-Objective Rewards**: Configurable reward shaping with curriculum learning support
- 🍎 **Apple Silicon Optimized**: Native MPS (Metal) acceleration detection
- 📊 **TensorBoard Integration**: Real-time training visualization
- 📦 **ONNX Export**: Deploy trained models to hardware

## Quick Start (macOS)

### 1. Install Dependencies (Recommended: Conda/Miniforge)
Due to PyBullet compilation issues on macOS Apple Silicon, we strongly recommend using Conda:

```bash
# 1. Install Miniforge (if not installed)
brew install miniforge
conda init zsh && source ~/.zshrc

# 2. Create environment with pre-compiled PyBullet
conda create -n robotics python=3.11 pybullet -c conda-forge -y
conda activate robotics

# 3. Install remaining dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Setup
Run the included verification script to check PyBullet, MPS acceleration, and environment:
```bash
python verify_setup.py
```

### 3. Training
```bash
# Fast headless training
robot-train train

# Live visualization (Watch Mode)
robot-train train --watch

# Custom parameters
robot-train train --algorithm sac --timesteps 1000000 --envs 8
```

### 4. Evaluation
```bash
# Evaluate trained model
robot-train eval --model outputs/checkpoints/latest/model.zip

# Visual evaluation
robot-train eval -m outputs/checkpoints/latest/model.zip --render
```

### 5. Export
```bash
# Export to ONNX for deployment
robot-train export --model outputs/checkpoints/latest/model.zip --optimize
```

## Project Structure
```
subconscious-robotics/
├── pyproject.toml          # Dependencies
├── configs/
│   ├── config.yaml         # Main Hydra config
│   ├── physics.yaml        # Domain randomization
│   ├── reward.yaml         # Reward shaping
│   └── agent.yaml          # PPO/SAC hyperparams
├── src/
│   ├── env/
│   │   ├── base_env.py            # Gymnasium environment
│   │   ├── domain_randomization.py # Physics randomization
│   │   ├── reward_shaper.py       # Multi-objective rewards
│   │   └── urdf_loader.py         # Robot loading utility
│   ├── models/
│   │   ├── device_utils.py        # MPS/CUDA detection
│   │   └── policy_networks.py     # MLP/CNN architectures
│   ├── train.py            # Training script
│   ├── eval.py             # Evaluation script
│   └── cli.py              # CLI interface
├── export/
│   └── onnx_export.py      # ONNX conversion
├── assets/
│   └── quadruped.urdf      # 8-DoF quadruped robot
└── outputs/                # Training artifacts
```

## Swapping Robot Designs

1. Place your URDF file in `assets/`

2. Update `configs/config.yaml`:
   ```yaml
   env:
     urdf_path: assets/your_robot.urdf
   ```

3. Or via CLI:
   ```bash
   robot-train train env.urdf_path=assets/your_robot.urdf
   ```

## Configuration Reference

### Domain Randomization (`configs/physics.yaml`)

```yaml
domain_randomization:
  mass:
    range: [0.85, 1.15]  # ±15% mass variation
  friction:
    lateral_range: [0.5, 1.5]
  gravity:
    range: [-10.5, -9.0]  # Gravity variation
```

### Reward Shaping (`configs/reward.yaml`)

```yaml
reward:
  weights:
    forward_velocity: 2.0
    stability: -1.0
    energy: -0.005
    survival: 0.1
```

### Agent Hyperparameters (`configs/agent.yaml`)

```yaml
ppo:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  gamma: 0.99
```

## TensorBoard

```bash
tensorboard --logdir outputs/logs
```

## Requirements

- Python 3.10+
- macOS (Apple Silicon recommended) / Linux / Windows
- PyBullet 3.2.5+
- Stable Baselines3 2.2.1+
- PyTorch 2.1+ (with MPS support)

## License

MIT
