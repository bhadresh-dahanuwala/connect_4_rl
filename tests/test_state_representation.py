"""Tests for state/observation representation."""

import numpy as np

from connect_4_rl.env import BOARD_COLS, BOARD_ROWS, Connect4Env


class TestObservationSpace:
    def test_observation_shape(self, env: Connect4Env) -> None:
        """Observation should be (3, 8, 8)."""
        obs, _ = env.reset()
        assert obs["observation"].shape == (3, BOARD_ROWS, BOARD_COLS)

    def test_observation_dtype(self, env: Connect4Env) -> None:
        """Observation should be int8."""
        obs, _ = env.reset()
        assert obs["observation"].dtype == np.int8

    def test_observation_values_binary(self, env: Connect4Env) -> None:
        """All observation values should be 0 or 1."""
        env.reset()
        for col in range(BOARD_COLS):
            obs, _, terminated, _, _ = env.step(col)
            observation = obs["observation"]
            assert np.all((observation == 0) | (observation == 1))
            if terminated:
                break


class TestPlaneRepresentation:
    def test_plane_0_red_coins(self, env: Connect4Env) -> None:
        """Plane 0 should accurately show red coin positions."""
        env.reset()
        env.step(0)  # Red at (7,0)
        obs, _, _, _, _ = env.step(1)  # Blue at (7,1)

        red_plane = obs["observation"][0]
        assert red_plane[7, 0] == 1
        assert np.sum(red_plane) == 1

    def test_plane_1_blue_coins(self, env: Connect4Env) -> None:
        """Plane 1 should accurately show blue coin positions."""
        env.reset()
        env.step(0)  # Red
        obs, _, _, _, _ = env.step(1)  # Blue at (7,1)

        blue_plane = obs["observation"][1]
        assert blue_plane[7, 1] == 1
        assert np.sum(blue_plane) == 1

    def test_plane_2_target_spaces_initial(self, env: Connect4Env) -> None:
        """Initially target plane should mark bottom row."""
        obs, _ = env.reset()
        target_plane = obs["observation"][2]

        for col in range(BOARD_COLS):
            assert target_plane[7, col] == 1
        assert np.sum(target_plane) == BOARD_COLS

    def test_plane_2_target_spaces_after_move(self, env: Connect4Env) -> None:
        """Target space should update after placing coin."""
        env.reset()
        env.step(0)
        obs, _, _, _, _ = env.step(0)  # Two coins in col 0

        target_plane = obs["observation"][2]
        assert target_plane[5, 0] == 1  # Next target at row 5
        assert target_plane[7, 0] == 0  # No longer at bottom
        assert target_plane[6, 0] == 0

    def test_planes_mutually_exclusive(self, env: Connect4Env) -> None:
        """Red and blue planes should never overlap."""
        env.reset()
        for _ in range(10):
            col = np.random.randint(0, BOARD_COLS)
            obs, _, terminated, _, _ = env.step(col)

            red_plane = obs["observation"][0]
            blue_plane = obs["observation"][1]
            overlap = red_plane & blue_plane
            assert np.sum(overlap) == 0

            if terminated:
                break

    def test_full_column_no_target(self, env: Connect4Env) -> None:
        """Full columns should have no target space."""
        env.reset()
        terminated = False

        # Fill column 0
        for i in range(BOARD_ROWS):
            obs, _, terminated, _, _ = env.step(0)
            if terminated:
                break
            if i < BOARD_ROWS - 1:
                obs, _, terminated, _, _ = env.step(1)
                if terminated:
                    break

        if not terminated:
            target_plane = obs["observation"][2]
            # Column 0 should have no target
            assert np.sum(target_plane[:, 0]) == 0


class TestMultipleMoves:
    def test_coin_count_increases(self, env: Connect4Env) -> None:
        """Total coins should increase with each move."""
        env.reset()

        for move_num in range(1, 6):
            obs, _, terminated, _, _ = env.step(move_num % BOARD_COLS)
            if terminated:
                break

            red_count = np.sum(obs["observation"][0])
            blue_count = np.sum(obs["observation"][1])
            total = red_count + blue_count

            assert total == move_num

    def test_observation_consistency(self, env: Connect4Env) -> None:
        """Observation should be consistent with internal state."""
        env.reset()
        for col in [0, 1, 2, 3, 0, 1, 2, 3]:
            obs, _, terminated, _, _ = env.step(col)
            if terminated:
                break

            # Verify observation matches board state
            for row in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    if env.board[row, c] == 1:
                        assert obs["observation"][0, row, c] == 1
                    elif env.board[row, c] == 2:
                        assert obs["observation"][1, row, c] == 1


class TestGetObservation:
    def test_get_observation_returns_current_state(
        self, env: Connect4Env
    ) -> None:
        """get_observation() should return current board state."""
        env.reset()
        env.step(0)  # Red
        env.step(1)  # Blue

        obs = env.get_observation()
        assert obs["observation"][0, 7, 0] == 1  # Red at (7,0)
        assert obs["observation"][1, 7, 1] == 1  # Blue at (7,1)

    def test_get_observation_matches_step_observation(
        self, env: Connect4Env
    ) -> None:
        """get_observation() should match observation from step()."""
        env.reset()
        step_obs, _, _, _, _ = env.step(3)
        get_obs = env.get_observation()

        assert np.array_equal(
            step_obs["observation"], get_obs["observation"]
        )
        assert np.array_equal(
            step_obs["action_mask"], get_obs["action_mask"]
        )

    def test_get_observation_after_terminal(self, env: Connect4Env) -> None:
        """Both players can see terminal state via get_observation()."""
        env.reset()
        # Create a vertical win for Red
        for i in range(3):
            env.step(0)  # Red
            env.step(1)  # Blue

        # Red wins
        obs, reward, terminated, _, info = env.step(0)
        assert terminated is True

        # Opponent can query terminal state
        opponent_obs = env.get_observation()
        assert np.array_equal(obs["observation"], opponent_obs["observation"])
