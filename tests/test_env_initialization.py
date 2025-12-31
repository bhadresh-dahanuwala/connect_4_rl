"""Tests for environment initialization and reset."""

import numpy as np

from connect_4_rl.env import BOARD_COLS, BOARD_ROWS, PLAYER_RED, Connect4Env


class TestInitialization:
    def test_board_dimensions(self, env: Connect4Env) -> None:
        """Board should be 8x8."""
        obs, _ = env.reset()
        assert obs["observation"].shape == (3, BOARD_ROWS, BOARD_COLS)
        assert obs["observation"].shape == (3, 8, 8)

    def test_initial_board_empty(self, env: Connect4Env) -> None:
        """Board should start empty."""
        obs, _ = env.reset()
        assert np.all(obs["observation"][0] == 0)
        assert np.all(obs["observation"][1] == 0)

    def test_initial_target_spaces(self, env: Connect4Env) -> None:
        """Target plane should show bottom row initially."""
        obs, _ = env.reset()
        target_plane = obs["observation"][2]

        assert np.all(target_plane[7] == 1)
        assert np.all(target_plane[:7] == 0)

    def test_initial_action_mask(self, env: Connect4Env) -> None:
        """All columns should be valid initially."""
        obs, _ = env.reset()
        assert np.all(obs["action_mask"] == 1)

    def test_initial_player(self, env: Connect4Env) -> None:
        """Red player should start first."""
        _, info = env.reset()
        assert info["current_player"] == PLAYER_RED

    def test_action_space(self, env: Connect4Env) -> None:
        """Action space should be Discrete(8)."""
        assert env.action_space.n == 8

    def test_reset_clears_board(self, env: Connect4Env) -> None:
        """Reset should clear any previous state."""
        env.reset()
        env.step(0)
        obs, _ = env.reset()
        assert np.all(obs["observation"][0] == 0)
        assert np.all(obs["observation"][1] == 0)

    def test_reset_with_seed(self, env: Connect4Env) -> None:
        """Reset should accept a seed parameter."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        assert np.array_equal(obs1["observation"], obs2["observation"])
