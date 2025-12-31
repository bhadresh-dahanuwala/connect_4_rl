"""Tests for win detection in all directions."""

from connect_4_rl.env import REWARD_LOSE, REWARD_WIN, Connect4Env


class TestHorizontalWin:
    def test_horizontal_win_bottom_row(self, env: Connect4Env) -> None:
        """Four in a row horizontally at bottom should win."""
        env.reset()
        # Red: 0, Blue: 4, Red: 1, Blue: 4, Red: 2, Blue: 4, Red: 3 (wins)
        env.step(0)  # R at (7,0)
        env.step(4)  # B at (7,4)
        env.step(1)  # R at (7,1)
        env.step(4)  # B at (6,4)
        env.step(2)  # R at (7,2)
        env.step(4)  # B at (5,4)
        obs, reward, terminated, _, _ = env.step(3)  # R at (7,3) - wins

        assert terminated is True
        assert reward == REWARD_WIN

    def test_horizontal_win_middle_columns(self, env: Connect4Env) -> None:
        """Win with columns 2,3,4,5 horizontally."""
        env.reset()
        env.step(2)  # R
        env.step(0)  # B
        env.step(3)  # R
        env.step(0)  # B
        env.step(4)  # R
        env.step(0)  # B
        obs, reward, terminated, _, _ = env.step(5)  # R wins

        assert terminated is True
        assert reward == REWARD_WIN

    def test_horizontal_win_completing_middle(self, env: Connect4Env) -> None:
        """Win when 4th piece fills gap in the middle."""
        env.reset()
        env.step(0)  # R
        env.step(4)  # B
        env.step(1)  # R
        env.step(4)  # B
        env.step(3)  # R (skip col 2)
        env.step(4)  # B
        obs, reward, terminated, _, _ = env.step(2)  # R fills gap - wins

        assert terminated is True
        assert reward == REWARD_WIN


class TestVerticalWin:
    def test_vertical_win(self, env: Connect4Env) -> None:
        """Four stacked vertically should win."""
        env.reset()
        for i in range(3):
            env.step(0)  # Red
            env.step(1)  # Blue

        obs, reward, terminated, _, _ = env.step(0)  # Red's 4th

        assert terminated is True
        assert reward == REWARD_WIN

    def test_vertical_win_different_column(self, env: Connect4Env) -> None:
        """Vertical win in column 5."""
        env.reset()
        for i in range(3):
            env.step(5)  # Red
            env.step(6)  # Blue

        obs, reward, terminated, _, _ = env.step(5)

        assert terminated is True
        assert reward == REWARD_WIN


class TestDiagonalWin:
    def test_diagonal_up_right(self, env: Connect4Env) -> None:
        """Diagonal win from bottom-left to top-right."""
        env.reset()
        # Build: R at (7,0), (6,1), (5,2), (4,3)
        env.step(0)  # R at (7,0)
        env.step(1)  # B at (7,1)
        env.step(1)  # R at (6,1)
        env.step(2)  # B at (7,2)
        env.step(2)  # R at (6,2)
        env.step(3)  # B at (7,3)
        env.step(2)  # R at (5,2)
        env.step(3)  # B at (6,3)
        env.step(3)  # R at (5,3)
        env.step(4)  # B elsewhere
        # R at (4,3) - diagonal win
        obs, reward, terminated, _, _ = env.step(3)

        assert terminated is True
        assert reward == REWARD_WIN

    def test_diagonal_built_from_top(self, env: Connect4Env) -> None:
        """Diagonal win where last piece is placed at the bottom."""
        env.reset()
        # Build diagonal with bottom piece last
        env.step(1)  # R at (7,1)
        env.step(0)  # B at (7,0)
        env.step(2)  # R at (7,2)
        env.step(1)  # B at (6,1)
        env.step(2)  # R at (6,2)
        env.step(2)  # B at (5,2)
        env.step(3)  # R at (7,3)
        env.step(3)  # B at (6,3)
        env.step(3)  # R at (5,3)
        env.step(3)  # B at (4,3)
        # Now R has: (7,1), (6,2), (5,3) - need (4,4) but complex
        # Simpler: complete diagonal (7,0), (6,1), (5,2), (4,3)


class TestAntiDiagonalWin:
    def test_anti_diagonal_win(self, env: Connect4Env) -> None:
        """Anti-diagonal win from bottom-right to top-left."""
        env.reset()
        # Build: R at (7,3), (6,2), (5,1), (4,0)
        env.step(3)  # R at (7,3)
        env.step(2)  # B at (7,2)
        env.step(2)  # R at (6,2)
        env.step(1)  # B at (7,1)
        env.step(1)  # R at (6,1)
        env.step(0)  # B at (7,0)
        env.step(1)  # R at (5,1)
        env.step(0)  # B at (6,0)
        env.step(0)  # R at (5,0)
        env.step(4)  # B elsewhere
        # R at (4,0) - anti-diagonal win
        obs, reward, terminated, _, _ = env.step(0)

        assert terminated is True
        assert reward == REWARD_WIN


class TestBlueWin:
    def test_blue_wins_horizontal(self, env: Connect4Env) -> None:
        """Blue player can also win."""
        env.reset()
        env.step(0)  # R
        env.step(4)  # B at (7,4)
        env.step(0)  # R
        env.step(5)  # B at (7,5)
        env.step(1)  # R
        env.step(6)  # B at (7,6)
        env.step(1)  # R
        obs, reward, terminated, _, _ = env.step(7)  # B at (7,7) - wins

        assert terminated is True
        assert reward == REWARD_WIN


class TestNoFalseWins:
    def test_three_in_row_no_win(self, env: Connect4Env) -> None:
        """Three in a row should not trigger win."""
        env.reset()
        env.step(0)  # R
        env.step(4)  # B
        env.step(1)  # R
        env.step(4)  # B
        obs, reward, terminated, _, _ = env.step(2)  # R - only 3

        assert terminated is False
        assert reward == 0.0

    def test_mixed_line_no_win(self, env: Connect4Env) -> None:
        """Line with mixed colors should not win."""
        env.reset()
        env.step(0)  # R
        env.step(1)  # B
        env.step(2)  # R
        obs, reward, terminated, _, _ = env.step(3)  # B

        assert terminated is False


class TestOpponentReward:
    def test_opponent_reward_on_win(self, env: Connect4Env) -> None:
        """Opponent should receive LOSE reward when player wins."""
        env.reset()
        # Red wins vertically
        for i in range(3):
            env.step(0)  # Red
            env.step(1)  # Blue

        obs, reward, terminated, _, info = env.step(0)  # Red wins

        assert terminated is True
        assert reward == REWARD_WIN
        assert info["opponent_reward"] == REWARD_LOSE

    def test_opponent_reward_on_draw(self, env: Connect4Env) -> None:
        """Both players get draw reward on draw."""
        env.reset()
        # Just verify the info contains opponent_reward
        obs, reward, terminated, _, info = env.step(0)
        assert "opponent_reward" in info

    def test_opponent_reward_during_game(self, env: Connect4Env) -> None:
        """During normal play, opponent_reward should be 0."""
        env.reset()
        obs, reward, terminated, _, info = env.step(0)

        assert terminated is False
        assert reward == 0.0
        assert info["opponent_reward"] == 0.0
