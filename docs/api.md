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

## Agents & Training

### AlphaZeroAgent

```python
from connect_4_rl.agent import AlphaZeroAgent
```

An agent that uses MCTS guided by a neural network to select moves.

#### Constructor

```python
AlphaZeroAgent(
    model: torch.nn.Module,
    num_simulations: int = 100,
    c_puct: float = 1.0,
    device: str = 'cpu'
)
```

**Parameters:**

- `model`: The neural network instance (`Connect4Net`)
- `num_simulations`: Number of MCTS simulations per move
- `c_puct`: Exploration constant for MCTS
- `device`: Device to run inference on ('cpu' or 'cuda'/'mps')

#### select_move()

```python
select_move(
    env: Connect4Env,
    temperature: float = 1.0,
    add_noise: bool = False
) -> tuple[int, np.ndarray]
```

Select a move using the MCTS policy.

**Parameters:**

- `env`: The current game environment
- `temperature`: Controls exploration (higher = more random)
- `add_noise`: If `True`, adds Dirichlet noise to root priors (for exploration)

**Returns:**

- `action`: Selected column index
- `action_probs`: Probability distribution over actions (policy)

---

### MCTS

```python
from connect_4_rl.mcts import MCTS
```

Monte Carlo Tree Search implementation for AlphaZero.

#### Constructor

```python
MCTS(
    model: torch.nn.Module,
    num_simulations: int = 100,
    c_puct: float = 1.0,
    device: str = 'cpu'
)
```

#### search()

```python
search(env: Connect4Env, add_noise: bool = False) -> Node
```

Performs MCTS simulations from the current state.

**Returns:**

- `root`: The root node of the search tree containing visit counts and values.

#### get_action_probs()

```python
get_action_probs(
    env: Connect4Env,
    temperature: float = 1.0,
    add_noise: bool = False
) -> np.ndarray
```

Runs MCTS and returns the action probability distribution based on visit counts.

---

### Connect4Net

```python
from connect_4_rl.model import Connect4Net
```

Residual Neural Network for the AlphaZero agent.

#### Constructor

```python
Connect4Net(num_channels: int = 128, num_res_blocks: int = 10)
```

**Parameters:**

- `num_channels`: Number of filters in convolutional layers
- `num_res_blocks`: Number of residual blocks in the tower

#### forward()

```python
forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
```

**Input:**

- `x`: Tensor of shape `(Batch, 3, 8, 8)` representing the board state.

**Returns:**

- `policy`: Logits for the 8 column actions `(Batch, 8)`
- `value`: Win probability `(Batch, 1)` in range `[-1, 1]`

---

### Trainer

```python
from connect_4_rl.trainer import Trainer
```

Orchestrates the training process: Self-Play, Training, and Evaluation.

#### Constructor

```python
Trainer(args: dict)
```

**Parameters:**

- `args`: Dictionary containing configuration:
    - `iterations`: Total training iterations (default: 5000)
    - `num_self_play_games`: Games per iteration (default: 100)
    - `num_simulations`: MCTS simulations per move in self-play (default: 400)
    - `eval_simulations`: MCTS simulations per move in evaluation (default: 100)
    - `num_eval_games`: Evaluation games per iteration (default: 50)
    - `batch_size`: Training batch size (default: 512)
    - `epochs`: Training epochs per iteration (default: 3)
    - `lr`: Learning rate (default: 0.001)
    - `c_puct`: MCTS exploration constant (default: 2.0)
    - `num_channels`: Model channels (default: 128)
    - `num_blocks`: Model residual blocks (default: 20)
    - `max_buffer_size`: Replay buffer capacity (default: 50000)
    - `workers`: Number of parallel processes (default: 10)
    - `resume`: Path to checkpoint to resume from (default: None)

#### run()

```python
run()
```

Executes the main training loop. Saves checkpoints to `checkpoints/`.

The training follows a specific pattern:
- **Iteration 1**: M0 vs M0, Train(M0)→M1, Eval M0 vs M1, pick Best
- **Iteration 2**: Best vs Best, Train(Best)→M2, Eval Best vs M2
- **Iteration 3+**: M{n-1} vs Best, Train(M{n-1})→M{n}, Eval M{n} vs Best

New models must win ≥62% of evaluation games to become the new champion.

#### self_play()

```python
self_play(challenger_state: dict, champion_state: dict) -> list
```

Generates training examples by playing challenger vs champion.
Training examples are collected from the challenger's moves.

**Parameters:**

- `challenger_state`: State dict of the model being trained
- `champion_state`: State dict of the current best model

**Returns:**

- List of training examples: `[(board, action_probs, outcome), ...]`

#### evaluate()

```python
evaluate(challenger_state: dict, champion_state: dict) -> float
```

Evaluates a new model (challenger) against the current best (champion).

**Parameters:**

- `challenger_state`: State dict of the newly trained model
- `champion_state`: State dict of the current best model

**Returns:**

- Win ratio for the challenger: `(wins + 0.5 * draws) / total_games`

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