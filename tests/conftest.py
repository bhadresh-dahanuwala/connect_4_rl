"""Shared pytest fixtures for Connect-4 tests."""

import pytest

from connect_4_rl.env import Connect4Env


@pytest.fixture
def env() -> Connect4Env:
    """Create a fresh environment for each test."""
    return Connect4Env()


@pytest.fixture
def env_with_render() -> Connect4Env:
    """Create environment with rendering enabled."""
    return Connect4Env(render_mode="ansi")
