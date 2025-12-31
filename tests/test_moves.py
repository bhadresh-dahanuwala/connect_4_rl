"""Tests for valid and invalid moves."""

import pytest

from connect_4_rl.env import BOARD_ROWS, PLAYER_BLUE, PLAYER_RED, Connect4Env


class TestValidMoves:
    def test_first_move(self, env: Connect4Env) -> None:
        """First move should place coin at bottom of column."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        assert obs["observation"][0, 7, 0] == 1
        assert reward == 0.0
        assert terminated is False
        assert truncated is False

    def test_stacking_coins(self, env: Connect4Env) -> None:
        """Coins should stack on top of each other."""
        env.reset()
        env.step(3)  # Red at row 7, col 3
        env.step(3)  # Blue at row 6, col 3
        obs, _, _, _, _ = env.step(3)  # Red at row 5, col 3

        assert obs["observation"][0, 7, 3] == 1  # Red
        assert obs["observation"][1, 6, 3] == 1  # Blue
        assert obs["observation"][0, 5, 3] == 1  # Red

    def test_alternating_players(self, env: Connect4Env) -> None:
        """Players should alternate turns."""
        _, info = env.reset()
        assert info["current_player"] == PLAYER_RED

        _, _, _, _, info = env.step(0)
        assert info["current_player"] == PLAYER_BLUE

        _, _, _, _, info = env.step(1)
        assert info["current_player"] == PLAYER_RED

    def test_target_space_updates(self, env: Connect4Env) -> None:
        """Target space should move up after coin placement."""
        env.reset()
        obs, _, _, _, _ = env.step(4)

        target_plane = obs["observation"][2]
        assert target_plane[6, 4] == 1
        assert target_plane[7, 4] == 0

    def test_info_contains_last_move(self, env: Connect4Env) -> None:
        """Info should contain details about the last move."""
        env.reset()
        _, _, _, _, info = env.step(3)

        assert info["last_action"] == 3
        assert info["last_row"] == 7
        assert info["last_player"] == PLAYER_RED


class TestInvalidMoves:
    def test_full_column_raises_error(self, env: Connect4Env) -> None:
        """Dropping in a full column should raise ValueError."""
        env.reset()
        for i in range(BOARD_ROWS):
            env.step(0)
            if i < BOARD_ROWS - 1:
                env.step(1)

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(0)

    def test_full_column_mask_updates(self, env: Connect4Env) -> None:
        """Action mask should mark full columns as invalid."""
        env.reset()
        for i in range(BOARD_ROWS):
            obs, _, terminated, _, _ = env.step(2)
            if terminated:
                break
            if i < BOARD_ROWS - 1:
                obs, _, terminated, _, _ = env.step(3)
                if terminated:
                    break

        if not terminated:
            assert obs["action_mask"][2] == 0
            assert obs["action_mask"][3] == 1

    def test_out_of_range_action_high(self, env: Connect4Env) -> None:
        """Actions above 7 should be invalid."""
        env.reset()
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(8)

    def test_out_of_range_action_negative(self, env: Connect4Env) -> None:
        """Negative actions should be invalid."""
        env.reset()
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(-1)
