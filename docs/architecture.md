# Architecture

This document describes the overall system architecture and how components interact.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                           Trainer                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Self-Play  │───▶│   Training  │───▶│    Evaluation       │  │
│  │  (Parallel) │    │   (GPU)     │    │    (Parallel)       │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│         │                 │                      │               │
│         ▼                 ▼                      ▼               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Examples  │    │    Model    │    │   Win Ratio ≥62%?   │  │
│  │   Buffer    │    │   Weights   │    │   Update Champion   │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Hierarchy

```
connect_4_rl/
├── env.py          # Gymnasium environment
├── model.py        # Neural network (Connect4Net)
├── mcts.py         # Monte Carlo Tree Search
├── agent.py        # AlphaZeroAgent (wraps MCTS + Model)
└── trainer.py      # Training orchestration
```

## Components

### 1. Environment (`env.py`)

The `Connect4Env` class implements the Gymnasium interface for an 8x8 Connect-4 game.

**Responsibilities:**
- Manage board state and game rules
- Validate moves and detect wins
- Provide observations and action masks
- Support both players viewing terminal states (for self-play)

**Key Design Decisions:**
- 3-plane observation: (red coins, blue coins, target spaces)
- Target plane shows where coins would land (aids neural network learning)
- `opponent_reward` in info dict enables proper credit assignment in self-play

### 2. Neural Network (`model.py`)

The `Connect4Net` class is a residual neural network that outputs both policy and value.

```
Input (3, 8, 8)
    │
    ▼
┌─────────────────┐
│  Conv2d 3→128   │
│  BatchNorm      │
│  ReLU           │
└────────┬────────┘
         │
    ┌────▼────┐
    │ ResBlock │ ×20
    │ (skip)   │
    └────┬────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│Policy │ │ Value │
│ Head  │ │ Head  │
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
  (8,)      (1,)
```

**Policy Head:** Predicts probability distribution over 8 columns
**Value Head:** Predicts expected game outcome (-1 to +1)

### 3. Monte Carlo Tree Search (`mcts.py`)

MCTS uses the neural network to guide tree search, balancing exploration and exploitation.

**Algorithm:**
1. **Select**: Traverse tree using PUCT formula until leaf node
2. **Expand**: Use neural network to get prior probabilities for children
3. **Evaluate**: Use neural network value head to estimate position value
4. **Backup**: Propagate value back up the tree, flipping signs

**PUCT Formula:**
```
UCB(s,a) = Q(s,a) + c_puct × P(s,a) × sqrt(N(s)) / (1 + N(s,a))
```

Where:
- `Q(s,a)` = average value of action a from state s
- `P(s,a)` = prior probability from neural network
- `N(s)` = visit count of parent
- `N(s,a)` = visit count of this action
- `c_puct` = exploration constant (default: 2.0)

### 4. Agent (`agent.py`)

The `AlphaZeroAgent` wraps MCTS and provides a clean interface for move selection.

**Key Methods:**
- `select_move(env, temperature, add_noise)`: Returns action and policy

**Temperature:**
- `temperature=1.0`: Sample proportionally to visit counts (exploration)
- `temperature=0.0`: Select most-visited action (exploitation)

**Dirichlet Noise:**
- Added to root priors during self-play
- Encourages exploration of novel moves
- `noise = Dir(α)` where α=0.3 for Connect-4

### 5. Trainer (`trainer.py`)

Orchestrates the full AlphaZero training loop.

**Training Cycle:**
```
┌──────────────────────────────────────────────────────────┐
│                    Iteration N                            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. SELF-PLAY                                            │
│     ┌─────────────────────────────────────────────┐      │
│     │  Challenger (M{n-1}) vs Champion (Best)     │      │
│     │  100 games × 10 workers in parallel         │      │
│     │  Collect examples from Challenger's moves   │      │
│     └─────────────────────────────────────────────┘      │
│                          │                               │
│                          ▼                               │
│  2. TRAINING                                             │
│     ┌─────────────────────────────────────────────┐      │
│     │  Add examples to replay buffer              │      │
│     │  Train M{n-1} → M{n} on buffer              │      │
│     │  3 epochs, batch size 512                   │      │
│     └─────────────────────────────────────────────┘      │
│                          │                               │
│                          ▼                               │
│  3. EVALUATION                                           │
│     ┌─────────────────────────────────────────────┐      │
│     │  M{n} vs Champion (Best)                    │      │
│     │  50 games, 100 MCTS simulations             │      │
│     │  If win rate ≥ 62%: Best = M{n}             │      │
│     └─────────────────────────────────────────────┘      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Parallel Execution:**
- Self-play and evaluation use `ProcessPoolExecutor`
- Each worker runs on CPU with its own model copy
- Main process trains on GPU (MPS/CUDA)

## Data Flow

### Training Example Format

```python
(board_state, action_probs, outcome)
```

- `board_state`: (3, 8, 8) numpy array
- `action_probs`: (8,) numpy array from MCTS visit counts
- `outcome`: +1 (win), -1 (loss), 0 (draw)

### State Dictionary Format

Model weights are shared between processes as CPU tensors:

```python
state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
```

## Key Design Patterns

### 1. Challenger vs Champion

Unlike pure self-play, this implementation uses asymmetric training:
- **Challenger**: The model being improved (collects training data)
- **Champion**: The current best model (provides strong opposition)

Benefits:
- Prevents distribution shift (training on current skill level)
- Provides consistent difficulty during learning
- Clear success metric (beat the champion)

### 2. Separate Eval Simulations

Self-play uses 400 MCTS simulations, but evaluation uses only 100.

Rationale:
- High simulations mask policy differences (search compensates for weak policy)
- Lower simulations reveal raw neural network quality
- Ensures new champion has genuinely better policy, not just more search

### 3. Gated Updates (62% Threshold)

New models must decisively beat the champion (≥62% win rate).

Benefits:
- Prevents noise from causing regression
- Ensures each champion is meaningfully stronger
- 62% corresponds to ~2:1 win ratio (statistically significant)

### 4. Replay Buffer

Training uses a sliding window buffer of recent examples.

```python
examples = deque(maxlen=50000)
```

Benefits:
- Prevents catastrophic forgetting
- Maintains diversity of positions
- Limits memory usage

## File Locations

| Purpose | Location |
|---------|----------|
| Checkpoints | `checkpoints/checkpoint_{iter}.pt` |
| Best Model | `checkpoints/best_model.pt` |
| Training Log | `training.log` |
| Play GUI | `play.py` |
| Training Script | `train.py` |
