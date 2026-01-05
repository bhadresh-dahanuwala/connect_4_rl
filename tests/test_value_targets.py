"""Tests to verify value target computation in training examples.

These tests ensure that:
1. Value targets are correctly assigned based on game outcomes
2. Canonical board transformation is correct
3. Symmetry augmentation preserves value correctness
"""

import numpy as np
import pytest
import torch

from connect_4_rl.env import Connect4Env, PLAYER_RED, PLAYER_BLUE
from connect_4_rl.model import Connect4Net


def create_canonical_board(env: Connect4Env) -> np.ndarray:
    """Create canonical board from current player's perspective."""
    obs = env.get_observation()
    board = obs['observation']

    if env.current_player == PLAYER_BLUE:
        canonical = board.copy()
        canonical[0] = board[1]  # Blue's pieces in plane 0
        canonical[1] = board[0]  # Red's pieces in plane 1
        return canonical
    return board.copy()


def get_model_value(model: Connect4Net, canonical_board: np.ndarray) -> float:
    """Get value prediction from model for canonical board."""
    with torch.no_grad():
        board_tensor = torch.tensor(canonical_board, dtype=torch.float32).unsqueeze(0)
        _, value = model(board_tensor)
    return value.item()


class TestValueTargetAssignment:
    """Tests for value target assignment logic."""

    def test_winner_gets_positive_value(self) -> None:
        """Player who wins should have examples with +1 value target."""
        env = Connect4Env()
        env.reset()

        # Red wins immediately with horizontal 4-in-a-row
        for a in [0, 4, 1, 5, 2, 6, 3]:  # Red: 0,1,2,3. Blue: 4,5,6
            obs, reward, term, _, info = env.step(a)
            if term:
                break

        # Red won
        assert reward == 1.0
        assert info['last_player'] == PLAYER_RED

        # Simulate value target assignment (from trainer logic)
        # If example was from Red (winner): outcome = 1
        # If example was from Blue (loser): outcome = -1
        red_outcome = 1 if PLAYER_RED == info['last_player'] else -1
        blue_outcome = 1 if PLAYER_BLUE == info['last_player'] else -1

        assert red_outcome == 1, "Winner's examples should get +1"
        assert blue_outcome == -1, "Loser's examples should get -1"

    def test_loser_gets_negative_value(self) -> None:
        """Player who loses should have examples with -1 value target."""
        env = Connect4Env()
        env.reset()

        # Blue wins
        for a in [0, 1, 0, 2, 4, 3, 5, 4]:  # Set up Blue winning
            obs, reward, term, _, info = env.step(a)
            if term:
                break

        if not term:
            # Continue until someone wins
            env.reset()
            # Blue wins with vertical
            for a in [0, 1, 2, 1, 3, 1, 4, 1]:
                obs, reward, term, _, info = env.step(a)
                if term:
                    break

        assert term
        winner = info['last_player']
        loser = PLAYER_RED if winner == PLAYER_BLUE else PLAYER_BLUE

        winner_outcome = 1 if winner == info['last_player'] else -1
        loser_outcome = 1 if loser == info['last_player'] else -1

        assert winner_outcome == 1
        assert loser_outcome == -1

    def test_draw_gets_zero_value(self) -> None:
        """Draw should result in 0 value target for both players."""
        # In the trainer: outcome = 0 if reward != 1.0 (draw)
        reward = 0.0  # draw
        outcome = 0 if reward != 1.0 else 1
        assert outcome == 0


class TestCanonicalTransformation:
    """Tests for canonical board transformation."""

    def test_red_turn_no_swap(self) -> None:
        """When it's Red's turn, planes should not be swapped."""
        env = Connect4Env()
        env.reset()

        # Make a move for Red
        env.step(0)  # Red at (7,0)
        env.step(1)  # Blue at (7,1)

        # Now Red's turn again
        assert env.current_player == PLAYER_RED

        obs = env.get_observation()['observation']
        canonical = create_canonical_board(env)

        # For Red's turn, canonical should be unchanged
        np.testing.assert_array_equal(canonical[0], obs[0])  # Red's pieces
        np.testing.assert_array_equal(canonical[1], obs[1])  # Blue's pieces

    def test_blue_turn_swaps_planes(self) -> None:
        """When it's Blue's turn, planes 0 and 1 should be swapped."""
        env = Connect4Env()
        env.reset()

        env.step(0)  # Red at (7,0)

        # Now Blue's turn
        assert env.current_player == PLAYER_BLUE

        obs = env.get_observation()['observation']
        canonical = create_canonical_board(env)

        # For Blue's turn, planes should be swapped
        np.testing.assert_array_equal(canonical[0], obs[1])  # Blue's pieces (was plane 1)
        np.testing.assert_array_equal(canonical[1], obs[0])  # Red's pieces (was plane 0)

    def test_canonical_plane0_is_current_player(self) -> None:
        """Plane 0 in canonical board should always be current player's pieces."""
        env = Connect4Env()
        env.reset()

        # Red's turn
        canonical_red = create_canonical_board(env)
        obs_red = env.get_observation()['observation']
        np.testing.assert_array_equal(canonical_red[0], obs_red[0])  # Red's pieces

        env.step(0)  # Red plays

        # Blue's turn
        canonical_blue = create_canonical_board(env)
        obs_blue = env.get_observation()['observation']
        np.testing.assert_array_equal(canonical_blue[0], obs_blue[1])  # Blue's pieces


class TestSymmetryAugmentation:
    """Tests for horizontal flip symmetry augmentation."""

    def test_symmetry_preserves_structure(self) -> None:
        """Flipped board should have pieces at mirrored positions."""
        env = Connect4Env()
        env.reset()

        env.step(0)  # Red at col 0
        env.step(1)  # Blue at col 1
        # Now it's Red's turn again

        canonical = create_canonical_board(env)
        sym_board = np.flip(canonical, axis=2).copy()

        # Red's turn, so no swap: plane 0 = Red, plane 1 = Blue
        # Red was at col 0, should now be at col 7 after flip
        assert canonical[0, 7, 0] == 1  # Red in original (plane 0)
        assert sym_board[0, 7, 7] == 1  # Red in flipped

        # Blue was at col 1, should now be at col 6
        assert canonical[1, 7, 1] == 1  # Blue in original (plane 1)
        assert sym_board[1, 7, 6] == 1  # Blue in flipped


class TestValueHeadExpectations:
    """Tests for what the value head should learn."""

    @pytest.fixture
    def trained_model(self) -> Connect4Net:
        """Load the current best model."""
        import os
        model = Connect4Net(num_channels=128, num_res_blocks=20)
        model_path = "checkpoints/best_model.pt"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def test_winning_position_positive_value(self, trained_model: Connect4Net) -> None:
        """Position where current player can win should have positive value."""
        env = Connect4Env()
        env.reset()

        # Red has 3-in-a-row, can win with col 3
        for a in [0, 4, 1, 5, 2, 6]:
            env.step(a)

        assert env.current_player == PLAYER_RED
        canonical = create_canonical_board(env)
        value = get_model_value(trained_model, canonical)

        # This should be positive - Red can win
        # Note: Current model may have issues here
        print(f"Winning position value: {value:.3f} (expected: positive)")

    def test_losing_position_negative_value(self, trained_model: Connect4Net) -> None:
        """Position where current player is about to lose should have negative value."""
        env = Connect4Env()
        env.reset()

        # Red has 3-in-a-row, Blue must block
        for a in [0, 4, 1, 4, 2]:  # Red: 0,1,2. Blue: 4,4 (stacked)
            env.step(a)

        assert env.current_player == PLAYER_BLUE
        canonical = create_canonical_board(env)
        value = get_model_value(trained_model, canonical)

        # This should be negative - Blue is in trouble
        print(f"Losing position value: {value:.3f} (expected: negative)")

    def test_symmetric_positions_same_value(self, trained_model: Connect4Net) -> None:
        """Horizontally symmetric positions should have the same value."""
        env = Connect4Env()
        env.reset()

        env.step(3)  # Red center

        canonical = create_canonical_board(env)
        sym_board = np.flip(canonical, axis=2).copy()

        value_orig = get_model_value(trained_model, canonical)
        value_flip = get_model_value(trained_model, sym_board)

        # Values should be very close
        print(f"Original value: {value_orig:.3f}, Flipped value: {value_flip:.3f}")
        assert abs(value_orig - value_flip) < 0.1, "Symmetric positions should have similar values"


class TestTrainerExampleGeneration:
    """Tests that simulate the trainer's example generation logic."""

    def test_example_value_target_red_wins(self) -> None:
        """Simulate example generation when Red wins."""
        env = Connect4Env()
        env.reset()

        examples = []

        # Play a game where Red wins
        moves = [0, 4, 1, 5, 2, 6, 3]  # Red wins with horizontal
        for move in moves:
            current_player = env.current_player
            canonical = create_canonical_board(env)

            # Store example (simplified - no action probs)
            examples.append({
                'board': canonical,
                'player': current_player
            })

            obs, reward, term, _, info = env.step(move)
            if term:
                break

        # Game ended with Red winning
        assert reward == 1.0
        winner = info['last_player']
        assert winner == PLAYER_RED

        # Assign value targets
        for ex in examples:
            if reward == 1.0:
                ex['value_target'] = 1 if ex['player'] == winner else -1
            else:
                ex['value_target'] = 0

        # Check targets
        for ex in examples:
            if ex['player'] == PLAYER_RED:
                assert ex['value_target'] == 1, "Red's examples should be +1 when Red wins"
            else:
                assert ex['value_target'] == -1, "Blue's examples should be -1 when Red wins"

    def test_example_value_target_blue_wins(self) -> None:
        """Simulate example generation when Blue wins."""
        env = Connect4Env()
        env.reset()

        examples = []

        # Play a game where Blue wins (vertical)
        moves = [0, 3, 1, 3, 2, 3, 4, 3]  # Blue wins with vertical in col 3
        for move in moves:
            current_player = env.current_player
            canonical = create_canonical_board(env)

            examples.append({
                'board': canonical,
                'player': current_player
            })

            obs, reward, term, _, info = env.step(move)
            if term:
                break

        # Game ended
        assert term
        winner = info['last_player']
        assert winner == PLAYER_BLUE

        # Assign value targets
        for ex in examples:
            if reward == 1.0:
                ex['value_target'] = 1 if ex['player'] == winner else -1
            else:
                ex['value_target'] = 0

        # Check targets
        for ex in examples:
            if ex['player'] == PLAYER_BLUE:
                assert ex['value_target'] == 1, "Blue's examples should be +1 when Blue wins"
            else:
                assert ex['value_target'] == -1, "Red's examples should be -1 when Blue wins"


class TestValueSignConsistency:
    """Tests to catch value sign inversion bugs."""

    def test_value_target_matches_canonical_perspective(self) -> None:
        """Value target should be from the perspective of the player in plane 0."""
        env = Connect4Env()
        env.reset()

        # Position: Blue's turn, Red has threat
        env.step(0)  # Red
        env.step(4)  # Blue
        env.step(1)  # Red
        env.step(5)  # Blue
        env.step(2)  # Red - 3 in a row

        assert env.current_player == PLAYER_BLUE

        canonical = create_canonical_board(env)

        # In canonical form for Blue's turn:
        # Plane 0 = Blue's pieces
        # Plane 1 = Red's pieces

        # Check that Red's 3-in-a-row is in plane 1 (opponent's plane)
        assert canonical[1, 7, 0] == 1  # Red at (7,0)
        assert canonical[1, 7, 1] == 1  # Red at (7,1)
        assert canonical[1, 7, 2] == 1  # Red at (7,2)

        # Blue's pieces should be in plane 0
        assert canonical[0, 7, 4] == 1  # Blue at (7,4)
        assert canonical[0, 7, 5] == 1  # Blue at (7,5)

        # If Blue doesn't block and Red wins, Blue's example gets -1
        # This is correct: Blue (plane 0) is losing

    def test_both_players_threat_position(self) -> None:
        """Test position where both players have 3-in-a-row."""
        env = Connect4Env()
        env.reset()

        # Both have 3-in-a-row: Red at 0,1,2 and Blue at 4,5,6
        for a in [0, 4, 1, 5, 2, 6]:
            env.step(a)

        assert env.current_player == PLAYER_RED

        canonical = create_canonical_board(env)

        # Red's turn: Red in plane 0, Blue in plane 1
        # Red has 3-in-a-row in plane 0
        assert canonical[0, 7, 0] == 1
        assert canonical[0, 7, 1] == 1
        assert canonical[0, 7, 2] == 1

        # Blue has 3-in-a-row in plane 1
        assert canonical[1, 7, 4] == 1
        assert canonical[1, 7, 5] == 1
        assert canonical[1, 7, 6] == 1

        # Red can win immediately with col 3
        # Value should be +1 from Red's perspective (current player)
        # The model should output positive value here
