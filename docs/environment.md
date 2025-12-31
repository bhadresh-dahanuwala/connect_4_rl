# Environment Specification

## Overview

The Connect-4 environment implements an 8x8 board variant of the classic
Connect-4 game. Two players (Red and Blue) take turns dropping coins into
columns, attempting to connect 4 coins in a row.

## Board

- **Dimensions**: 8 rows x 8 columns
- **Win Condition**: 4 consecutive coins (horizontal, vertical, or diagonal)
- **Row Indexing**: Row 0 is the top, Row 7 is the bottom
- **Column Indexing**: Columns 0-7 from left to right

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
7 . . . . . . .    <- Coins land here first
```

## Players

| Player | ID | Symbol | Turn Order |
|--------|----| -------|------------|
| Red    | 1  | R      | First      |
| Blue   | 2  | B      | Second     |

## Action Space

- **Type**: `Discrete(8)`
- **Values**: 0-7 (column index to drop coin)
- **Invalid Actions**: Columns that are full (8 coins already placed)

Use the `action_mask` to determine valid actions:

```python
action_mask = obs["action_mask"]  # Shape: (8,)
valid_actions = [i for i in range(8) if action_mask[i] == 1]
```

## Observation Space

The observation is a dictionary with two keys:

### `observation` - Board State

- **Shape**: `(3, 8, 8)`
- **Type**: `int8`
- **Values**: Binary (0 or 1)

| Plane | Description                                    |
|-------|------------------------------------------------|
| 0     | Red coins (1 where Red coin exists)            |
| 1     | Blue coins (1 where Blue coin exists)          |
| 2     | Target spaces (1 where next coin would land)   |

**Plane 2 (Target Spaces)**: Shows the landing position for each column.
If a column is full, its target space is all zeros.

### `action_mask` - Valid Actions

- **Shape**: `(8,)`
- **Type**: `int8`
- **Values**: Binary (0 = invalid, 1 = valid)

## Rewards

Rewards are returned from the perspective of the player who just moved.

| Outcome | Reward | `opponent_reward` |
|---------|--------|-------------------|
| Win     | +1.0   | -1.0              |
| Loss    | N/A    | N/A               |
| Draw    | 0.0    | 0.0               |
| Step    | 0.0    | 0.0               |

The `opponent_reward` is available in the `info` dictionary, allowing both
players to receive their rewards in self-play scenarios.

## Episode Termination

The episode terminates when:

1. **Win**: A player connects 4 coins in a row (any direction)
2. **Draw**: The board is full (64 coins placed) with no winner

The `truncated` flag is always `False` (no time limits).

## Info Dictionary

The `info` dictionary returned by `step()` contains:

| Key               | Type  | Description                              |
|-------------------|-------|------------------------------------------|
| `current_player`  | int   | Player to move next (1=Red, 2=Blue)      |
| `last_player`     | int   | Player who just moved                    |
| `last_action`     | int   | Column of the last move                  |
| `last_row`        | int   | Row where the last coin landed           |
| `opponent_reward` | float | Reward for the opponent (for self-play)  |

## Win Detection

Wins are checked in 4 directions from the last placed coin:

1. **Horizontal**: Left-right along the same row
2. **Vertical**: Up-down along the same column
3. **Diagonal**: Top-left to bottom-right
4. **Anti-diagonal**: Top-right to bottom-left

A win requires exactly 4 consecutive coins of the same color.
