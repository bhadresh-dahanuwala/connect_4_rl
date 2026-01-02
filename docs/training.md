# Training Guide

This guide explains how to train the AlphaZero agent to play Connect-4.

## Overview

The training process follows the AlphaZero algorithm, iteratively improving the agent through self-play and reinforcement learning.

The core components are:
1.  **Self-Play**: The agent plays against itself to generate training data.
2.  **Training**: The neural network is trained on the generated data to predict moves (policy) and game outcomes (value).
3.  **Evaluation**: The new model is evaluated against the previous version to track progress.

## The Trainer

The `Trainer` class orchestrates the entire process. It manages parallel workers for self-play and evaluation, handles model checkpointing, and controls the learning rate.

### Configuration

The trainer is configured via a dictionary of arguments:

| Argument | Description | Typical Value |
|----------|-------------|---------------|
| `iterations` | Number of training cycles | 100+ |
| `num_self_play_games` | Games generated per iteration | 100-500 |
| `num_simulations` | MCTS simulations per move | 50-200 |
| `num_channels` | CNN filter channels | 64-128 |
| `num_blocks` | Residual blocks in NN | 5-10 |
| `batch_size` | Training batch size | 64-256 |
| `epochs` | Training epochs per iteration | 10 |
| `lr` | Learning rate | 0.001 |
| `workers` | Number of parallel CPU workers | CPU Core Count |

## Self-Play Process

During self-play, the agent plays games against itself to collect experience.

1.  **Opening Diversity**: For the first 5 moves, the agent chooses randomly from valid actions. This ensures a diverse set of opening positions are explored.
2.  **MCTS Search**: After move 5, the agent uses Monte Carlo Tree Search (MCTS) to select moves.
3.  **Temperature Decay**:
    *   The "temperature" controls the exploration in move selection.
    *   It starts at 1.0 (probabilistic selection based on visit counts) and decays to 0.1 over time.
    *   This encourages exploration early in the game and stronger play later.
4.  **Symmetries**: The board is symmetrical (horizontal flip). The trainer automatically augments the data by flipping boards and policies, effectively doubling the training data.

## Model Architecture

The agent uses a Residual Neural Network (`Connect4Net`) with:
*   **Input**: 3 planes (Current Player, Opponent, Target/Valid)
*   **Body**: Stack of Residual Blocks (Conv2D + BatchNorm + ReLU)
*   **Policy Head**: Outputs probabilities for the 8 columns.
*   **Value Head**: Outputs expected win probability (-1 to 1).

## Training Loop

1.  **Generate Data**: `num_self_play_games` are played in parallel.
2.  **Replay Buffer**: New games are added to a sliding window buffer (`max_buffer_size`).
3.  **Optimize**: The network is trained on batches sampled from the buffer.
    *   **Loss**: Sum of Mean Squared Error (Value) and Cross-Entropy (Policy).
    *   **Scheduler**: Learning rate is reduced if training loss plateaus.
4.  **Checkpoint**: The model is saved to `checkpoints/`.

## Evaluation

At the end of each iteration, the new model plays against the previous iteration's model.
*   **Metric**: Win Ratio (Wins + 0.5 * Draws).
*   **Logging**: Detailed stats on wins, losses, draws, and average game length.
*   **Note**: In this implementation, we always accept the new model for the next iteration (no gating), but we track performance to monitor regression.
