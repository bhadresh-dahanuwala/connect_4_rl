#!/usr/bin/env python3
"""Diagnose if the neural network is actually learning."""

import argparse
import glob
import os
import re
import torch
import numpy as np
from connect_4_rl.env import Connect4Env, PLAYER_RED, PLAYER_BLUE
from connect_4_rl.model import Connect4Net


def get_latest_checkpoint():
    """Find the latest checkpoint by iteration number."""
    pattern = "checkpoints/checkpoint_*.pt"
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None

    def get_iter(path):
        match = re.search(r'checkpoint_(\d+)\.pt$', path)
        return int(match.group(1)) if match else -1

    checkpoints.sort(key=get_iter, reverse=True)
    return checkpoints[0] if checkpoints else None


def load_model(path="checkpoints/best_model.pt", num_blocks=20, num_channels=128):
    model = Connect4Net(num_channels=num_channels, num_res_blocks=num_blocks)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {path}")
    except FileNotFoundError:
        print(f"No model found at {path}, using random weights")
    model.eval()
    return model


def get_raw_policy(model, env):
    """Get raw policy output from neural network (no MCTS)."""
    obs = env.get_observation()
    board = obs["observation"]

    # Canonical form (current player's perspective)
    if env.current_player == PLAYER_BLUE:
        canonical = board.copy()
        canonical[0] = board[1]
        canonical[1] = board[0]
    else:
        canonical = board

    with torch.no_grad():
        board_tensor = torch.tensor(canonical, dtype=torch.float32).unsqueeze(0)
        policy_logits, value = model(board_tensor)
        policy = torch.softmax(policy_logits, dim=1).squeeze().numpy()
        value = value.item()

    return policy, value


def print_policy(policy, action_mask, label=""):
    """Print policy distribution."""
    print(f"\n{label}")
    print("Column:     ", "   ".join(f"{i}" for i in range(8)))
    print("Policy:     ", "  ".join(f"{p:.0%}" if m else " -- " for p, m in zip(policy, action_mask)))
    print(f"Max prob:    {policy.max():.1%} (column {policy.argmax()})")
    print(f"Entropy:     {-np.sum(policy * np.log(policy + 1e-8)):.2f} (uniform=2.08)")


def test_opening():
    """Test raw policy on opening position."""
    print("\n" + "="*60)
    print("TEST 1: Opening Position")
    print("="*60)

    model = load_model()
    env = Connect4Env()
    env.reset()

    policy, value = get_raw_policy(model, env)
    action_mask = env.get_observation()["action_mask"]

    print_policy(policy, action_mask, "Raw policy (no MCTS):")
    print(f"Value prediction: {value:.3f}")

    # Check if policy is uniform (not learning) or concentrated (learning)
    max_prob = policy.max()
    if max_prob < 0.20:
        print("\n⚠️  Policy is nearly uniform - network may not be learning")
    else:
        print(f"\n✓ Policy shows preference (max {max_prob:.0%})")


def test_obvious_threat():
    """Test if raw policy recognizes an obvious blocking move."""
    print("\n" + "="*60)
    print("TEST 2: Must-Block Position")
    print("="*60)
    print("Red has 3-in-a-row at bottom. Blue must block column 3.")

    model = load_model()
    env = Connect4Env()
    env.reset()

    # Create position: Red has coins at columns 0,1,2 (bottom row)
    # Blue plays elsewhere. Red threatens win at column 3.
    env.step(0)  # Red
    env.step(0)  # Blue (stacks)
    env.step(1)  # Red
    env.step(1)  # Blue (stacks)
    env.step(2)  # Red - now has 3 in a row!
    # Blue's turn - MUST play column 3 to block

    print("\nBoard state:")
    env.render()

    policy, value = get_raw_policy(model, env)
    action_mask = env.get_observation()["action_mask"]

    print_policy(policy, action_mask, "Blue's raw policy (no MCTS):")
    print(f"Value prediction: {value:.3f} (should be negative, Blue is in trouble)")

    col3_prob = policy[3]
    if col3_prob > 0.5:
        print(f"\n✓ Network correctly prioritizes blocking move (col 3: {col3_prob:.0%})")
    elif col3_prob > 0.2:
        print(f"\n~ Network somewhat recognizes threat (col 3: {col3_prob:.0%})")
    else:
        print(f"\n⚠️  Network does NOT see the threat! (col 3: {col3_prob:.0%})")
        print("   This suggests the policy is not learning patterns.")


def test_winning_move():
    """Test if raw policy recognizes a winning move."""
    print("\n" + "="*60)
    print("TEST 3: Winning Move Position")
    print("="*60)
    print("Red has 3-in-a-row. Red should play column 3 to win.")

    model = load_model()
    env = Connect4Env()
    env.reset()

    # Create position: Red has coins at columns 0,1,2 (bottom row)
    # Set up so it's Red's turn with winning move available
    env.step(0)  # Red
    env.step(4)  # Blue (plays elsewhere)
    env.step(1)  # Red
    env.step(5)  # Blue (plays elsewhere)
    env.step(2)  # Red - now has 3 in a row!
    env.step(6)  # Blue (plays elsewhere)
    # Red's turn - can win with column 3

    print("\nBoard state:")
    env.render()

    policy, value = get_raw_policy(model, env)
    action_mask = env.get_observation()["action_mask"]

    print_policy(policy, action_mask, "Red's raw policy (no MCTS):")
    print(f"Value prediction: {value:.3f} (should be positive, Red is winning)")

    col3_prob = policy[3]
    if col3_prob > 0.5:
        print(f"\n✓ Network correctly prioritizes winning move (col 3: {col3_prob:.0%})")
    elif col3_prob > 0.2:
        print(f"\n~ Network somewhat sees the win (col 3: {col3_prob:.0%})")
    else:
        print(f"\n⚠️  Network does NOT see the winning move! (col 3: {col3_prob:.0%})")


def compare_with_random():
    """Compare trained model policy with random weights."""
    print("\n" + "="*60)
    print("TEST 4: Trained vs Random Weights")
    print("="*60)

    trained = load_model()
    random_model = Connect4Net(num_channels=128, num_res_blocks=20)
    random_model.eval()

    env = Connect4Env()
    env.reset()

    trained_policy, trained_value = get_raw_policy(trained, env)

    # Temporarily swap to get random policy
    with torch.no_grad():
        board = env.get_observation()["observation"]
        board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
        random_logits, random_val = random_model(board_tensor)
        random_policy = torch.softmax(random_logits, dim=1).squeeze().numpy()

    action_mask = env.get_observation()["action_mask"]

    print("\nOpening position comparison:")
    print_policy(trained_policy, action_mask, "Trained model:")
    print_policy(random_policy, action_mask, "Random weights:")

    # Check if they're different
    diff = np.abs(trained_policy - random_policy).max()
    if diff < 0.1:
        print(f"\n⚠️  Trained model is very similar to random! (max diff: {diff:.1%})")
    else:
        print(f"\n✓ Trained model differs from random (max diff: {diff:.1%})")


def test_value_head():
    """Test if value head predicts correctly for clear positions."""
    print("\n" + "="*60)
    print("TEST 5: Value Head Analysis")
    print("="*60)

    model = load_model()
    env = Connect4Env()

    print("\nTesting value predictions across different positions:\n")

    # Test 1: Empty board
    env.reset()
    _, value = get_raw_policy(model, env)
    print(f"Empty board (Red to play):     value = {value:+.3f}  (expected: ~0)")

    # Test 2: Red about to win (3 in a row)
    env.reset()
    env.step(0); env.step(4)  # Red, Blue
    env.step(1); env.step(5)  # Red, Blue
    env.step(2); env.step(6)  # Red has 3 in row, Blue played elsewhere
    _, value = get_raw_policy(model, env)
    print(f"Red has 3-in-row, Red's turn:  value = {value:+.3f}  (expected: ~+1, can win)")

    # Test 3: Blue about to lose (Red has 3 in a row)
    env.reset()
    env.step(0); env.step(0)  # Red, Blue stacks
    env.step(1); env.step(1)  # Red, Blue stacks
    env.step(2)               # Red has 3 in row
    # Now it's Blue's turn, must block or lose
    _, value = get_raw_policy(model, env)
    print(f"Red has 3-in-row, Blue's turn: value = {value:+.3f}  (expected: ~-0.5, must block)")

    # Test 4: Near end of game (many pieces)
    env.reset()
    moves = [0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7]  # Fill bottom 3 rows alternating
    for m in moves:
        try:
            obs, reward, term, _, _ = env.step(m)
            if term:
                break
        except:
            break
    if not term:
        _, value = get_raw_policy(model, env)
        print(f"Mid-game (24 moves in):        value = {value:+.3f}  (expected: varies)")

    # Test 5: One move from Red winning
    env.reset()
    env.step(3); env.step(0)  # Red center, Blue corner
    env.step(3); env.step(1)  # Red stacks, Blue
    env.step(3); env.step(2)  # Red has vertical 3, Blue
    # Red can win with column 3
    _, value = get_raw_policy(model, env)
    print(f"Red 1 move from win (vertical):value = {value:+.3f}  (expected: ~+1)")

    # Check if value varies at all
    env.reset()
    values = []
    for col in range(8):
        env_copy = Connect4Env()
        env_copy.reset()
        env_copy.step(col)
        _, v = get_raw_policy(model, env_copy)
        values.append(v)

    print(f"\nValue after each opening move (cols 0-7):")
    print(f"  {['%.3f' % v for v in values]}")
    print(f"  Range: {max(values) - min(values):.4f}")

    if max(values) - min(values) < 0.01:
        print("\n⚠️  Value head outputs nearly constant values!")
        print("   It's not learning to distinguish positions.")
    else:
        print(f"\n✓ Value head shows some variation ({max(values)-min(values):.3f} range)")


def test_gradient_flow():
    """Check if gradients flow properly through the network."""
    print("\n" + "="*60)
    print("TEST 6: Gradient Flow Check")
    print("="*60)

    model = load_model()
    model.train()  # Enable gradient computation

    env = Connect4Env()
    env.reset()
    board = env.get_observation()["observation"]
    board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
    board_tensor.requires_grad = True

    policy_logits, value = model(board_tensor)

    # Check policy gradient
    policy_loss = policy_logits.sum()
    policy_loss.backward(retain_graph=True)

    policy_grad = board_tensor.grad.abs().mean().item()
    print(f"Policy gradient magnitude: {policy_grad:.6f}")

    # Check value gradient
    board_tensor.grad = None
    value_loss = value.sum()
    value_loss.backward()

    value_grad = board_tensor.grad.abs().mean().item()
    print(f"Value gradient magnitude:  {value_grad:.6f}")

    if policy_grad < 1e-6:
        print("\n⚠️  Policy gradients are essentially zero!")
    if value_grad < 1e-6:
        print("\n⚠️  Value gradients are essentially zero!")

    # Check weight magnitudes
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        zero_params += (param.abs() < 1e-6).sum().item()

    print(f"\nModel has {total_params:,} parameters")
    print(f"Near-zero parameters: {zero_params:,} ({100*zero_params/total_params:.1f}%)")


def analyze_training_data():
    """Analyze what the training data looks like."""
    print("\n" + "="*60)
    print("TEST 7: Training Data Analysis (Simulated)")
    print("="*60)

    print("\nSimulating one game with random policy to see training targets...")

    model = load_model()
    model.eval()
    env = Connect4Env()
    obs, info = env.reset()

    policies_collected = []
    values_collected = []

    # Play a game collecting what MCTS would produce
    from connect_4_rl.mcts import MCTS
    mcts = MCTS(model, num_simulations=100, c_puct=2.0, device='cpu')

    step = 0
    while True:
        action_probs = mcts.get_action_probs(env, temperature=1.0, add_noise=True)
        policies_collected.append(action_probs)

        # Get value estimate
        _, value = get_raw_policy(model, env)
        values_collected.append(value)

        # Select action
        action = np.random.choice(8, p=action_probs)
        obs, reward, terminated, _, info = env.step(action)
        step += 1

        if terminated:
            break

    print(f"Game length: {step} moves")
    print(f"Result: {'Win' if reward == 1.0 else 'Draw'}")

    print("\nPolicy distributions from MCTS (first 5 moves):")
    for i, p in enumerate(policies_collected[:5]):
        entropy = -np.sum(p * np.log(p + 1e-8))
        max_prob = p.max()
        print(f"  Move {i+1}: max={max_prob:.1%}, entropy={entropy:.2f} (uniform=2.08)")

    avg_entropy = np.mean([-np.sum(p * np.log(p + 1e-8)) for p in policies_collected])
    print(f"\nAverage policy entropy: {avg_entropy:.2f} (uniform=2.08)")

    if avg_entropy > 1.9:
        print("⚠️  MCTS policies are nearly uniform - no learning signal!")
        print("   This is because the neural network policy is uniform,")
        print("   so MCTS explores all moves roughly equally.")
    else:
        print("✓ MCTS policies have structure for learning")

    print(f"\nValue predictions during game: {[f'{v:.2f}' for v in values_collected[:5]]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose neural network learning")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model checkpoint (default: auto-detect latest or best)"
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="Use latest checkpoint instead of best_model.pt"
    )
    args = parser.parse_args()

    # Determine which model to test
    if args.model:
        model_path = args.model
    elif args.latest:
        model_path = get_latest_checkpoint()
        if not model_path:
            print("No checkpoints found, falling back to best_model.pt")
            model_path = "checkpoints/best_model.pt"
    else:
        model_path = "checkpoints/best_model.pt"

    print(f"\n{'='*60}")
    print(f"DIAGNOSING: {model_path}")
    print(f"{'='*60}")

    # Override the default path in all test functions
    import functools
    original_load = load_model
    load_model = functools.partial(original_load, path=model_path)

    test_opening()
    test_obvious_threat()
    test_winning_move()
    compare_with_random()
    test_value_head()
    test_gradient_flow()
    analyze_training_data()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("If the policy is uniform or doesn't recognize threats/wins,")
    print("the neural network is not learning - MCTS is doing all the work.")
    print("="*60)
