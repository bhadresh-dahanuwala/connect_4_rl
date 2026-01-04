# Training Guide

This guide explains how to train the AlphaZero agent to play Connect-4.

## Overview

The training process follows the AlphaZero algorithm, iteratively improving the agent through self-play and reinforcement learning.

The core components are:
1.  **Self-Play**: The challenger model plays against the champion to generate training data.
2.  **Training**: The neural network is trained on the generated data to predict moves (policy) and game outcomes (value).
3.  **Evaluation**: The trained model is evaluated against the champion; if it wins ≥62% of games, it becomes the new champion.

## Running Training

```bash
# Basic training run
poetry run python train.py

# Resume from checkpoint
poetry run python train.py --resume checkpoints/checkpoint_49.pt

# Custom configuration
poetry run python train.py --iterations 1000 --self-play-games 200 --simulations 800
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--iterations` | 5000 | Number of training iterations |
| `--self-play-games` | 100 | Games generated per iteration |
| `--epochs` | 3 | Training epochs per iteration |
| `--eval-games` | 50 | Evaluation games per iteration |
| `--simulations` | 400 | MCTS simulations per move (self-play) |
| `--eval-simulations` | 100 | MCTS simulations per move (evaluation) |
| `--cpuct` | 2.0 | MCTS exploration constant |
| `--workers` | 10 | Parallel worker processes |
| `--max-buffer-size` | 50000 | Replay buffer capacity |
| `--batch-size` | 512 | Training batch size |
| `--lr` | 0.001 | Learning rate |
| `--num-blocks` | 20 | Residual blocks in neural network |
| `--num-channels` | 128 | Channels in neural network |
| `--resume` | None | Path to checkpoint to resume from |

## Training Loop

The training follows a specific iteration pattern:

### Iteration 1 (Initial)
```
M0 = Random Weights
Self Play = M0 vs M0
M1 = Train(M0)
Eval = M0 vs M1
Best = M0 if M0 wins more, else M1
```

### Iteration 2
```
Self Play = Best vs Best
M2 = Train(Best)
Eval = Best vs M2
if M2 win rate >= 62%: Best = M2
```

### Iteration 3+ (Main Loop)
```
Self Play = M{n-1} vs Best
M{n} = Train(M{n-1})
Eval = M{n} vs Best
if M{n} win rate >= 62%: Best = M{n}
```

Key points:
- **Challenger vs Champion**: Self-play pits the model being trained (challenger) against the current best (champion)
- **Examples from Challenger**: Training data is collected from the challenger's moves
- **Gated Updates**: New models must win ≥62% against the champion to become the new best

## Self-Play Process

During self-play, the challenger plays against the champion to collect training experience.

1.  **Temperature-based Exploration**:
    - First 20 moves: temperature=1.0 with Dirichlet noise (exploration)
    - After move 20: temperature=0.0 (exploitation)

2.  **MCTS Search**: Each move uses Monte Carlo Tree Search with 400 simulations (configurable).

3.  **Symmetries**: The board is horizontally symmetric. Training data is automatically augmented by flipping boards and policies.

4.  **Parallel Execution**: Games run in parallel across multiple worker processes.

## Model Architecture

The agent uses a Residual Neural Network (`Connect4Net`) with:
- **Input**: 3 planes (Red coins, Blue coins, Target spaces)
- **Body**: Stack of Residual Blocks (Conv2D + BatchNorm + ReLU)
- **Policy Head**: Outputs probabilities for the 8 columns
- **Value Head**: Outputs expected win probability (-1 to 1)

Default architecture:
- 20 residual blocks
- 128 channels per layer
- ~5.9 million parameters

## Replay Buffer

Training examples are stored in a sliding window buffer:
- Maximum size: 50,000 examples (configurable)
- Examples include: board state, action probabilities, game outcome
- Older examples are discarded as new ones arrive

## Learning Rate Schedule

The learning rate adapts during training:
- Initial LR: 0.001
- Scheduler: ReduceLROnPlateau (reduces by 0.5x when loss plateaus)
- Patience: 4 iterations without improvement
- Burn-in: First 20 iterations use fixed LR
- Minimum LR: 1e-6

## Evaluation

After each training iteration, the new model is evaluated against the champion:
- **Games**: 50 evaluation games (configurable)
- **MCTS Simulations**: 100 (lower than self-play to test raw policy quality)
- **Win Ratio**: (Wins + 0.5 × Draws) / Total Games
- **Threshold**: Model must achieve ≥62% win rate to become new champion

Evaluation uses lower MCTS simulations than self-play to better measure the neural network's policy quality rather than search strength.

## Checkpoints

Models are saved to `checkpoints/`:
- `checkpoint_{iteration}.pt`: Periodic saves (last 5 kept)
- `best_model.pt`: Current champion model

Checkpoint contents:
- Model weights
- Optimizer state
- Scheduler state
- Iteration number

## Monitoring Training

Training logs include:
- Self-play progress and results (P1/P2 wins, draws)
- Training loss (policy + value components)
- Evaluation results and win rates
- Learning rate changes
- Champion updates

Example log output:
```
[2026-01-03 21:39:07] ITERATION 1/5000
[2026-01-03 21:39:07] [Self-Play] Running 100 games (10 workers)...
[2026-01-03 21:45:32] [Self-Play] Done: 3200 examples, avg 32.1 steps/game
[2026-01-03 21:45:32] [Self-Play] Results: P1=48, P2=45, Draws=7
[2026-01-03 21:45:45] [Training] Loss: 2.1543 -> 1.8234 (V=0.4521, P=1.3713)
[2026-01-03 21:46:12] [Eval] New=32, Old=15, Draws=3 | Win Rate: 67% | Avg Steps: 28.4
[2026-01-03 21:46:12] [Summary] Loss=1.8234, LR=0.001000, Buffer=3200, Status=IMPROVED
[2026-01-03 21:46:12]   > New Champion! Saving best_model.pt (Win Rate: 67%)
```

## Tips for Training

1. **Start with fewer simulations** (100-200) for faster initial iterations, increase later
2. **Monitor value loss**: If it's not decreasing, the model isn't learning game outcomes
3. **Watch for new champions**: Regular champion updates indicate learning progress
4. **Use MPS/CUDA**: Training uses GPU acceleration when available
5. **Checkpoint frequently**: Resume from checkpoints if training is interrupted
