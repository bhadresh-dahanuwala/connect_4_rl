# API Reference

## Connect4Env

```python
from connect_4_rl import Connect4Env
```

A Gymnasium-compatible environment for Connect-4 with an 8x8 board.

### Constructor

```python
Connect4Env(render_mode: str | None = None)
```

**Parameters:**

- `render_mode`: Optional rendering mode
  - `None`: No rendering
  - `"human"`: Print board to console
  - `"ansi"`: Return board as string

**Example:**

```python
env = Connect4Env()
env_with_render = Connect4Env(render_mode="human")
```

---

### Methods

#### reset()

```python
reset(
    seed: int | None = None,
    options: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]
```

Reset the environment to initial state.

**Parameters:**

- `seed`: Random seed for reproducibility
- `options`: Additional options (unused)

**Returns:**

- `observation`: Dictionary with `"observation"` and `"action_mask"`
- `info`: Dictionary with `"current_player"`

**Example:**

```python
obs, info = env.reset(seed=42)
print(info["current_player"])  # 1 (Red)
```

---

#### step()

```python
step(action: int) -> tuple[
    dict[str, np.ndarray],  # observation
    float,                   # reward
    bool,                    # terminated
    bool,                    # truncated
    dict[str, Any]           # info
]
```

Execute one move in the environment.

**Parameters:**

- `action`: Column index (0-7) to drop the coin

**Returns:**

- `observation`: Dictionary with `"observation"` and `"action_mask"`
- `reward`: `+1.0` (win), `0.0` (draw/step)
- `terminated`: `True` if game ended
- `truncated`: Always `False`
- `info`: Dictionary with game state details

**Raises:**

- `ValueError`: If action is invalid (column full or out of range)

**Example:**

```python
obs, reward, terminated, truncated, info = env.step(3)

if terminated:
    if reward == 1.0:
        print("Player won!")
    else:
        print("Draw!")
```

---

#### get_observation()

```python
get_observation() -> dict[str, np.ndarray]
```

Get current observation without taking a step.

Useful for self-play where both players need to see the terminal state.

**Returns:**

- Dictionary with `"observation"` and `"action_mask"`

**Example:**

```python
# After game ends, opponent can see terminal state
obs, reward, terminated, _, info = env.step(action)

if terminated:
    opponent_view = env.get_observation()
    opponent_reward = info["opponent_reward"]
```

---

#### render()

```python
render() -> str | None
```

Render the current board state.

**Returns:**

- `None` if `render_mode` is `None`
- Board string if `render_mode` is `"human"` or `"ansi"`

**Example:**

```python
env = Connect4Env(render_mode="ansi")
env.reset()
env.step(3)
board_str = env.render()
print(board_str)
```

Output:

```
  0 1 2 3 4 5 6 7
  ---------------
0 . . . . . . . .
1 . . . . . . . .
2 . . . . . . . .
3 . . . . . . . .
4 . . . . . . . .
5 . . . . . . . .
6 . . . . . . . .
7 . . . R . . . .
```

---

### Attributes

| Attribute           | Type            | Description                    |
|---------------------|-----------------|--------------------------------|
| `action_space`      | `Discrete(8)`   | Action space definition        |
| `observation_space` | `Dict`          | Observation space definition   |
| `board`             | `np.ndarray`    | Current board state (8x8)      |
| `current_player`    | `int`           | Current player (1=Red, 2=Blue) |
| `column_heights`    | `np.ndarray`    | Number of coins in each column |
| `render_mode`       | `str` or `None` | Current render mode            |

---

## Constants

```python
from connect_4_rl.env import (
    BOARD_ROWS,    # 8
    BOARD_COLS,    # 8
    WIN_LENGTH,    # 4
    PLAYER_RED,    # 1
    PLAYER_BLUE,   # 2
    EMPTY,         # 0
    REWARD_WIN,    # 1.0
    REWARD_LOSE,   # -1.0
    REWARD_DRAW,   # 0.0
    REWARD_STEP,   # 0.0
)
```

| Constant      | Value | Description               |
|---------------|-------|---------------------------|
| `BOARD_ROWS`  | 8     | Number of rows            |
| `BOARD_COLS`  | 8     | Number of columns         |
| `WIN_LENGTH`  | 4     | Coins needed to win       |
| `PLAYER_RED`  | 1     | Red player identifier     |
| `PLAYER_BLUE` | 2     | Blue player identifier    |
| `EMPTY`       | 0     | Empty cell identifier     |
| `REWARD_WIN`  | 1.0   | Reward for winning        |
| `REWARD_LOSE` | -1.0  | Reward for losing         |
| `REWARD_DRAW` | 0.0   | Reward for draw           |
| `REWARD_STEP` | 0.0   | Reward for regular step   |
