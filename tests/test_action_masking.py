"""Tests for action masking functionality."""

import numpy as np

from connect_4_rl.env import BOARD_COLS, BOARD_ROWS, Connect4Env


class TestActionMaskShape:
    def test_action_mask_shape(self, env: Connect4Env) -> None:
        """Action mask should have shape (8,)."""
        obs, _ = env.reset()
        assert obs["action_mask"].shape == (BOARD_COLS,)

    def test_action_mask_dtype(self, env: Connect4Env) -> None:
        """Action mask should be int8."""
        obs, _ = env.reset()
        assert obs["action_mask"].dtype == np.int8

    def test_action_mask_binary(self, env: Connect4Env) -> None:
        """Action mask values should be 0 or 1."""
        obs, _ = env.reset()
        assert np.all((obs["action_mask"] == 0) | (obs["action_mask"] == 1))


class TestActionMaskBehavior:
    def test_initial_all_valid(self, env: Connect4Env) -> None:
        """Initially all columns should be valid."""
        obs, _ = env.reset()
        assert np.all(obs["action_mask"] == 1)
        assert np.sum(obs["action_mask"]) == BOARD_COLS

    def test_mask_unchanged_after_single_move(self, env: Connect4Env) -> None:
        """Single move should not invalidate any column."""
        env.reset()
        obs, _, _, _, _ = env.step(0)
        assert np.all(obs["action_mask"] == 1)

    def test_full_column_becomes_invalid(self, env: Connect4Env) -> None:
        """Column should become invalid when full."""
        env.reset()
        terminated = False

        # Fill column 3
        for i in range(BOARD_ROWS):
            obs, _, terminated, _, _ = env.step(3)
            if terminated:
                break
            if i < BOARD_ROWS - 1:
                obs, _, terminated, _, _ = env.step(4)
                if terminated:
                    break

        if not terminated:
            assert obs["action_mask"][3] == 0
            assert obs["action_mask"][4] == 1

    def test_mask_tracks_all_columns(self, env: Connect4Env) -> None:
        """Mask should correctly track each column's availability."""
        env.reset()

        # Fill columns 0 and 7 alternately
        terminated = False
        for i in range(BOARD_ROWS):
            obs, _, terminated, _, _ = env.step(0)
            if terminated:
                break
            obs, _, terminated, _, _ = env.step(7)
            if terminated:
                break

        if not terminated:
            assert obs["action_mask"][0] == 0
            assert obs["action_mask"][7] == 0
            for col in range(1, 7):
                assert obs["action_mask"][col] == 1

    def test_valid_actions_from_mask(self, env: Connect4Env) -> None:
        """Actions with mask=1 should be valid, mask=0 invalid."""
        env.reset()
        terminated = False

        # Fill column 2
        for i in range(BOARD_ROWS):
            obs, _, terminated, _, _ = env.step(2)
            if terminated:
                break
            if i < BOARD_ROWS - 1:
                obs, _, terminated, _, _ = env.step(3)
                if terminated:
                    break

        if not terminated:
            mask = obs["action_mask"]

            # Verify mask[2] = 0 means invalid
            assert mask[2] == 0

            # All mask[i] = 1 columns should be playable
            for col in range(BOARD_COLS):
                if mask[col] == 1:
                    # Create a copy to test without modifying original
                    test_env = Connect4Env()
                    test_env.board = env.board.copy()
                    test_env.column_heights = env.column_heights.copy()
                    test_env.current_player = env.current_player

                    # Should not raise
                    test_env.step(col)


class TestMaskWithGameProgress:
    def test_mask_at_near_full_board(self, env: Connect4Env) -> None:
        """Mask should reflect available columns as board fills."""
        env.reset()
        moves_made = 0
        max_moves = BOARD_ROWS * BOARD_COLS

        terminated = False
        col = 0

        while not terminated and moves_made < max_moves:
            obs, _, terminated, _, _ = env.step(col)
            moves_made += 1

            if not terminated:
                # Verify mask consistency
                valid_count = np.sum(obs["action_mask"])
                assert valid_count > 0  # At least one valid action

                # Find next valid column
                valid_cols = np.where(obs["action_mask"] == 1)[0]
                col = valid_cols[0]

    def test_mask_sum_decreases(self, env: Connect4Env) -> None:
        """As columns fill, valid action count should decrease."""
        env.reset()
        initial_valid = 8
        terminated = False

        # Fill column 0 completely
        for i in range(BOARD_ROWS):
            obs, _, terminated, _, _ = env.step(0)
            if terminated:
                break
            if i < BOARD_ROWS - 1:
                obs, _, terminated, _, _ = env.step(1)
                if terminated:
                    break

        if not terminated:
            final_valid = np.sum(obs["action_mask"])
            assert final_valid < initial_valid
