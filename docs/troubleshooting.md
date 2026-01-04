# Troubleshooting

Common issues and solutions when training or playing.

## Training Issues

### Model Not Improving

**Symptom**: After many iterations, no new champion is crowned.

**Possible Causes & Solutions**:

1. **Threshold too high**: The 62% win rate may be too strict
   ```bash
   # Check recent evaluation results
   grep "Win Rate" training.log | tail -20
   ```
   If win rates are consistently 50-60%, consider lowering the threshold in `trainer.py`.

2. **Learning rate too high/low**:
   ```bash
   # Check if LR is decaying
   grep "LR" training.log | tail -10
   ```
   If LR hasn't decayed and loss is unstable, reduce initial LR.

3. **Buffer too small**: Not enough position diversity
   - Increase `--max-buffer-size` to 100000+

4. **Not enough simulations**: MCTS not finding good moves
   - Increase `--simulations` to 800+

### Training Crashes with MPS Error

**Symptom**: `RuntimeError: _share_filename_: only available on CPU`

**Cause**: PyTorch multiprocessing cannot share MPS tensors between processes.

**Solution**: Ensure model states are moved to CPU before passing to workers:
```python
state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
```

This is already handled in the current codebase.

### Out of Memory

**Symptom**: Training crashes with memory errors.

**Solutions**:

1. Reduce batch size:
   ```bash
   poetry run python train.py --batch-size 256
   ```

2. Reduce buffer size:
   ```bash
   poetry run python train.py --max-buffer-size 25000
   ```

3. Reduce number of workers:
   ```bash
   poetry run python train.py --workers 4
   ```

### Training Too Slow

**Symptom**: Each iteration takes too long.

**Solutions**:

1. Reduce simulations (trades quality for speed):
   ```bash
   poetry run python train.py --simulations 200
   ```

2. Reduce games per iteration:
   ```bash
   poetry run python train.py --self-play-games 50
   ```

3. Increase workers (if CPU-bound):
   ```bash
   poetry run python train.py --workers 16
   ```

### Loss Not Decreasing

**Symptom**: Training loss stays flat or increases.

**Possible Causes**:

1. **Learning rate too high**: Loss oscillates wildly
   - Reduce `--lr` to 0.0001

2. **Batch size too small**: Noisy gradients
   - Increase `--batch-size` to 512 or 1024

3. **Corrupted buffer**: Bad examples accumulated
   - Start fresh training without `--resume`

## GUI Issues

### Pygame Window Not Appearing

**Symptom**: `play.py` runs but no window shows.

**Solutions**:

1. Check pygame installation:
   ```bash
   poetry run python -c "import pygame; pygame.init(); print('OK')"
   ```

2. On macOS, try running from terminal (not IDE):
   ```bash
   poetry run python play.py
   ```

### Model Not Found

**Symptom**: `No model found at checkpoints/best_model.pt`

**Cause**: Training hasn't completed iteration 1 yet.

**Solution**: Wait for first iteration to complete, or specify a checkpoint:
```bash
poetry run python play.py --model checkpoints/checkpoint_0.pt
```

### AI Playing Randomly

**Symptom**: AI makes seemingly random moves.

**Possible Causes**:

1. **Untrained model**: Using random weights
   - Train for more iterations

2. **Wrong architecture**: Model shape mismatch
   ```bash
   # Match training architecture
   poetry run python play.py --num-blocks 20 --num-channels 128
   ```

3. **Too few simulations**:
   ```bash
   poetry run python play.py --simulations 400
   ```

## Environment Issues

### Invalid Action Error

**Symptom**: `ValueError: Invalid action: column X is full`

**Cause**: Attempting to play in a full column.

**Solution**: Always check action mask before stepping:
```python
action_mask = obs["action_mask"]
valid_actions = [i for i, valid in enumerate(action_mask) if valid]
action = random.choice(valid_actions)
```

### Observation Shape Mismatch

**Symptom**: Neural network input shape errors.

**Expected shapes**:
- `obs["observation"]`: `(3, 8, 8)`
- `obs["action_mask"]`: `(8,)`

**Solution**: Ensure you're using the dict observation correctly:
```python
board = obs["observation"]  # Not just obs
board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
```

## Checkpoint Issues

### Cannot Resume Training

**Symptom**: Error when using `--resume`

**Possible Causes**:

1. **File not found**: Check the path exists
   ```bash
   ls -la checkpoints/
   ```

2. **Corrupted checkpoint**: Try an earlier checkpoint
   ```bash
   poetry run python train.py --resume checkpoints/checkpoint_48.pt
   ```

3. **Architecture mismatch**: Checkpoint was saved with different model size
   - Use matching `--num-blocks` and `--num-channels`

### Disk Space Full

**Symptom**: Cannot save checkpoints.

**Solution**: Old checkpoints are automatically cleaned (keeping last 5). If disk is still full:
```bash
# Manually remove old checkpoints
ls -t checkpoints/checkpoint_*.pt | tail -n +6 | xargs rm
```

## Getting Help

If issues persist:

1. Check the training log for detailed error messages:
   ```bash
   tail -100 training.log
   ```

2. Run tests to verify installation:
   ```bash
   poetry run pytest tests/ -v
   ```

3. Check Python and dependency versions:
   ```bash
   poetry run python --version
   poetry show
   ```
