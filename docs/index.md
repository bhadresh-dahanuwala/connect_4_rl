# Connect-4 RL Environment

A Gymnasium-compatible reinforcement learning environment for Connect-4 with
an 8x8 board, designed for self-play training.

## Features

- **8x8 Board**: Extended board size for more complex gameplay
- **Gymnasium Interface**: Compatible with standard RL libraries
- **Action Masking**: Built-in support for valid action filtering
- **Self-Play Ready**: Both players can observe terminal states
- **3-Plane State Representation**: Optimized for neural network input

## Documentation

- [Installation](installation.md) - Setup and dependencies
- [Environment Specification](environment.md) - Board, actions, rewards, state
- [Training Guide](training.md) - AlphaZero training loop and configuration
- [API Reference](api.md) - Complete API documentation
- [Examples](examples.md) - Usage examples and self-play training

## Quick Start

```python
from connect_4_rl import Connect4Env

env = Connect4Env()
obs, info = env.reset()

# Game loop
terminated = False
while not terminated:
    action_mask = obs["action_mask"]
    valid_actions = [i for i in range(8) if action_mask[i]]
    action = valid_actions[0]  # Choose first valid action

    obs, reward, terminated, truncated, info = env.step(action)

print(f"Game ended. Reward: {reward}")
```

## Requirements

- Python >= 3.13
- gymnasium >= 1.0.0
- numpy
- torch >= 2.0.0
