# Examples

## Basic Usage

### Creating and Resetting the Environment

```python
from connect_4_rl import Connect4Env

# Create environment
env = Connect4Env()

# Reset to initial state
obs, info = env.reset()

print(f"Board shape: {obs['observation'].shape}")  # (3, 8, 8)
print(f"Action mask: {obs['action_mask']}")        # [1, 1, 1, 1, 1, 1, 1, 1]
print(f"Current player: {info['current_player']}") # 1 (Red)
```

### Making Moves

```python
env.reset()

# Red plays column 3
obs, reward, terminated, truncated, info = env.step(3)
print(f"Reward: {reward}, Terminated: {terminated}")

# Blue plays column 4
obs, reward, terminated, truncated, info = env.step(4)
print(f"Current player: {info['current_player']}")  # 1 (Red's turn)
```

### Using Action Masks

```python
env.reset()

# Get valid actions
action_mask = obs["action_mask"]
valid_actions = [i for i in range(8) if action_mask[i] == 1]
print(f"Valid actions: {valid_actions}")

# Always use the mask to avoid invalid moves
import random
action = random.choice(valid_actions)
obs, reward, terminated, truncated, info = env.step(action)
```

---

## Random Play

### Two Random Players

```python
import random
from connect_4_rl import Connect4Env

env = Connect4Env()
obs, info = env.reset()

move_count = 0
while True:
    # Get valid actions
    action_mask = obs["action_mask"]
    valid_actions = [i for i in range(8) if action_mask[i] == 1]

    # Choose random valid action
    action = random.choice(valid_actions)
    obs, reward, terminated, truncated, info = env.step(action)
    move_count += 1

    if terminated:
        if reward == 1.0:
            winner = "Red" if info["last_player"] == 1 else "Blue"
            print(f"{winner} wins after {move_count} moves!")
        else:
            print(f"Draw after {move_count} moves!")
        break
```

---

## Self-Play Training

### Basic Self-Play Loop

```python
from connect_4_rl import Connect4Env
from connect_4_rl.env import PLAYER_RED, PLAYER_BLUE

env = Connect4Env()

def select_action(obs, player):
    """Placeholder for your policy."""
    action_mask = obs["action_mask"]
    valid_actions = [i for i in range(8) if action_mask[i] == 1]
    return valid_actions[0]  # Replace with your policy

def train_step(obs, action, reward, next_obs, terminated):
    """Placeholder for your training logic."""
    pass

# Self-play episode
obs, info = env.reset()
player_obs = {PLAYER_RED: obs, PLAYER_BLUE: None}

while True:
    current_player = info["current_player"]

    # Select action for current player
    action = select_action(obs, current_player)

    # Take step
    next_obs, reward, terminated, truncated, info = env.step(action)

    if terminated:
        # Winner's update
        winner = info["last_player"]
        train_step(obs, action, reward, next_obs, terminated)

        # Loser's update - they also see the terminal state
        loser = PLAYER_BLUE if winner == PLAYER_RED else PLAYER_RED
        loser_reward = info["opponent_reward"]
        train_step(player_obs[loser], None, loser_reward, next_obs, terminated)
        break

    # Store observation for opponent's training
    opponent = PLAYER_BLUE if current_player == PLAYER_RED else PLAYER_RED
    player_obs[opponent] = next_obs

    obs = next_obs
```

### Collecting Experience for Both Players

```python
from connect_4_rl import Connect4Env
from connect_4_rl.env import PLAYER_RED, PLAYER_BLUE

def collect_episode(env, policy_red, policy_blue):
    """Collect experience for both players from a single game."""
    obs, info = env.reset()

    experiences = {
        PLAYER_RED: [],
        PLAYER_BLUE: []
    }

    policies = {
        PLAYER_RED: policy_red,
        PLAYER_BLUE: policy_blue
    }

    while True:
        current_player = info["current_player"]
        policy = policies[current_player]

        # Select action
        action = policy(obs)

        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Store experience
        experiences[current_player].append({
            "obs": obs,
            "action": action,
            "reward": reward if terminated else 0.0,
            "next_obs": next_obs,
            "terminated": terminated
        })

        if terminated:
            # Update opponent's last experience with their reward
            opponent = PLAYER_BLUE if current_player == PLAYER_RED else PLAYER_RED
            if experiences[opponent]:
                experiences[opponent][-1]["reward"] = info["opponent_reward"]
                experiences[opponent][-1]["next_obs"] = next_obs
                experiences[opponent][-1]["terminated"] = True
            break

        obs = next_obs

    return experiences
```

---

## Observation Inspection

### Visualizing Board State

```python
from connect_4_rl import Connect4Env
import numpy as np

env = Connect4Env(render_mode="ansi")
env.reset()

# Make some moves
env.step(3)  # Red
env.step(3)  # Blue
env.step(4)  # Red

# Render the board
print(env.render())

# Inspect observation planes
obs = env.get_observation()
red_plane = obs["observation"][0]
blue_plane = obs["observation"][1]
target_plane = obs["observation"][2]

print(f"\nRed coins at: {np.argwhere(red_plane == 1)}")
print(f"Blue coins at: {np.argwhere(blue_plane == 1)}")
print(f"Target spaces at: {np.argwhere(target_plane == 1)}")
```

### State Representation for Neural Networks

```python
import torch
from connect_4_rl import Connect4Env

env = Connect4Env()
obs, _ = env.reset()

# Convert observation to tensor
state = torch.tensor(obs["observation"], dtype=torch.float32)
print(f"State shape: {state.shape}")  # torch.Size([3, 8, 8])

# Add batch dimension for neural network
state_batch = state.unsqueeze(0)
print(f"Batch shape: {state_batch.shape}")  # torch.Size([1, 3, 8, 8])

# Action mask as tensor
action_mask = torch.tensor(obs["action_mask"], dtype=torch.bool)
```

---

## Testing Win Conditions

### Creating Specific Board States

```python
from connect_4_rl import Connect4Env

env = Connect4Env(render_mode="human")
env.reset()

# Create a horizontal win for Red
# Red plays: 0, 1, 2, 3 (bottom row)
# Blue plays: 0, 1, 2 (stacking on Red's coins)

env.step(0)  # Red at (7,0)
env.step(0)  # Blue at (6,0)
env.step(1)  # Red at (7,1)
env.step(1)  # Blue at (6,1)
env.step(2)  # Red at (7,2)
env.step(2)  # Blue at (6,2)

# Red wins with column 3
obs, reward, terminated, _, info = env.step(3)

env.render()
print(f"\nTerminated: {terminated}")
print(f"Winner reward: {reward}")
print(f"Loser reward: {info['opponent_reward']}")
```

---

## Reproducibility

### Using Seeds

```python
from connect_4_rl import Connect4Env
import numpy as np

env = Connect4Env()

# Reset with seed for reproducibility
obs1, _ = env.reset(seed=42)

# Reset again with same seed
obs2, _ = env.reset(seed=42)

# Observations should be identical
assert np.array_equal(obs1["observation"], obs2["observation"])
print("Reproducible reset confirmed!")
```
