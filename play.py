#!/usr/bin/env python3
"""Play Connect-4 against the trained AlphaZero agent using Pygame."""

import argparse
import os
import sys

import pygame
import torch


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller bundle."""
    if hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller bundle
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

from connect_4_rl.agent import AlphaZeroAgent
from connect_4_rl.env import Connect4Env, PLAYER_BLUE, PLAYER_RED
from connect_4_rl.model import Connect4Net

# --- Constants ---
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 900
BOARD_ROWS = 8
BOARD_COLS = 8
SQUARE_SIZE = 100
RADIUS = int(SQUARE_SIZE / 2 - 5)
OFFSET_X = (WINDOW_WIDTH - (BOARD_COLS * SQUARE_SIZE)) // 2
OFFSET_Y = (WINDOW_HEIGHT - (BOARD_ROWS * SQUARE_SIZE)) // 2 + 50

# Colors (Dark Room Friendly - Warm Tones)
BG_COLOR = (30, 28, 26)           # Dark warm grey
BOARD_COLOR = (50, 45, 40)        # Warm dark brown
EMPTY_COLOR = (25, 22, 20)        # Very dark brown
RED_COIN = (180, 70, 60)          # Muted warm red
YELLOW_COIN = (200, 150, 50)      # Muted amber/gold
TEXT_COLOR = (200, 190, 175)      # Warm off-white
HIGHLIGHT_COLOR = (120, 110, 100) # Warm grey
LAST_MOVE_COLOR = (220, 200, 170) # Warm cream for last move
WIN_HIGHLIGHT_COLOR = (220, 180, 80)  # Gold/amber for winning chips
BUTTON_COLOR = (80, 60, 50)       # Warm brown button
BUTTON_HOVER_COLOR = (100, 75, 60)  # Lighter warm brown on hover


class Connect4GUI:
    def __init__(self, model_path, simulations, c_puct, ai_first):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Connect-4: Human vs AlphaZero")
        self.font = pygame.font.SysFont("monospace", 40)
        self.small_font = pygame.font.SysFont("monospace", 25)
        self.button_font = pygame.font.SysFont("monospace", 20)

        # Store settings for restart
        self.model_path = model_path
        self.c_puct = c_puct
        self.ai_starts = ai_first

        # Initialize Game & Agent
        self.env = Connect4Env()
        self.obs, self.info = self.env.reset()

        self.device = 'cpu'
        self.model = Connect4Net().to(self.device)
        self.load_model(model_path)

        self.agent = AlphaZeroAgent(self.model, simulations, c_puct, self.device)
        self.simulations = simulations

        # Player Setup
        self._setup_players()

        self.game_over = False
        self.winner = None
        self.ai_thinking = False
        self.last_move = None  # (row, col) of last played chip
        self.winning_positions = []  # List of (row, col) for winning line
        self._update_message()

        # Button rectangles
        button_y = OFFSET_Y + BOARD_ROWS * SQUARE_SIZE + 20
        self.restart_button = pygame.Rect(OFFSET_X, button_y, 150, 40)
        self.switch_button = pygame.Rect(OFFSET_X + BOARD_COLS * SQUARE_SIZE - 200, button_y, 200, 40)

    def _setup_players(self):
        """Setup player assignments based on who starts first."""
        self.ai_player = PLAYER_RED if self.ai_starts else PLAYER_BLUE
        self.human_player = PLAYER_BLUE if self.ai_starts else PLAYER_RED

    def _update_message(self):
        """Update the status message based on game state."""
        if self.game_over:
            return  # Don't update if game is over
        if self.info['current_player'] == self.ai_player:
            self.message = "AI is thinking..."
        else:
            self.message = "Your Turn"

    def load_model(self, model_path):
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
        else:
            print(f"No model found at {model_path}. Using random initialization.")

    def restart_game(self):
        """Reset the game to initial state."""
        self.obs, self.info = self.env.reset()
        self._setup_players()
        self.game_over = False
        self.winner = None
        self.ai_thinking = False
        self.last_move = None
        self.winning_positions = []
        self._update_message()

    def toggle_first_player(self):
        """Toggle who starts first and restart the game."""
        self.ai_starts = not self.ai_starts
        self.restart_game()

    def draw_board(self):
        self.screen.fill(BG_COLOR)

        # Draw status text
        label = self.font.render(self.message, 1, TEXT_COLOR)
        self.screen.blit(label, (WINDOW_WIDTH // 2 - label.get_width() // 2, 20))

        # Draw Grid
        board = self.env.board
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                # Draw Board Rect
                pygame.draw.rect(
                    self.screen, BOARD_COLOR,
                    (c * SQUARE_SIZE + OFFSET_X, r * SQUARE_SIZE + OFFSET_Y, SQUARE_SIZE, SQUARE_SIZE)
                )

                # Draw Circle (Coin or Empty)
                color = EMPTY_COLOR
                if board[r][c] == PLAYER_RED:
                    color = RED_COIN
                elif board[r][c] == PLAYER_BLUE:
                    color = YELLOW_COIN

                center_x = int(c * SQUARE_SIZE + SQUARE_SIZE / 2 + OFFSET_X)
                center_y = int(r * SQUARE_SIZE + SQUARE_SIZE / 2 + OFFSET_Y)

                pygame.draw.circle(self.screen, color, (center_x, center_y), RADIUS)

                # Highlight winning positions with green ring
                if (r, c) in self.winning_positions:
                    pygame.draw.circle(
                        self.screen, WIN_HIGHLIGHT_COLOR,
                        (center_x, center_y), RADIUS, 4
                    )
                # Highlight last move with white ring (if not part of winning line)
                elif self.last_move == (r, c):
                    pygame.draw.circle(
                        self.screen, LAST_MOVE_COLOR,
                        (center_x, center_y), RADIUS, 3
                    )

        # Highlight column under mouse
        if not self.game_over and not self.ai_thinking:
            pos_x = pygame.mouse.get_pos()[0]
            if OFFSET_X <= pos_x <= OFFSET_X + BOARD_COLS * SQUARE_SIZE:
                col = int((pos_x - OFFSET_X) // SQUARE_SIZE)
                pygame.draw.rect(
                    self.screen, HIGHLIGHT_COLOR,
                    (col * SQUARE_SIZE + OFFSET_X, OFFSET_Y, SQUARE_SIZE, BOARD_ROWS * SQUARE_SIZE),
                    3
                )

        # Draw buttons
        mouse_pos = pygame.mouse.get_pos()

        # Restart button
        restart_color = BUTTON_HOVER_COLOR if self.restart_button.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(self.screen, restart_color, self.restart_button, border_radius=5)
        restart_text = self.button_font.render("Restart", True, TEXT_COLOR)
        self.screen.blit(restart_text, (
            self.restart_button.centerx - restart_text.get_width() // 2,
            self.restart_button.centery - restart_text.get_height() // 2
        ))

        # Switch first player button
        switch_color = BUTTON_HOVER_COLOR if self.switch_button.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(self.screen, switch_color, self.switch_button, border_radius=5)
        first_player = "AI" if self.ai_starts else "You"
        switch_text = self.button_font.render(f"First: {first_player}", True, TEXT_COLOR)
        self.screen.blit(switch_text, (
            self.switch_button.centerx - switch_text.get_width() // 2,
            self.switch_button.centery - switch_text.get_height() // 2
        ))

        pygame.display.update()

    def handle_ai_move(self):
        """Run AI search in a separate thread/process to keep UI responsive."""
        self.ai_thinking = True
        self.message = "AI is thinking..."
        self.draw_board()

        # Run MCTS
        # Note: Pygame doesn't like threads touching rendering, but this is pure logic
        action, _ = self.agent.select_move(self.env, temperature=0.0, add_noise=False)

        self.obs, reward, terminated, _, self.info = self.env.step(action)

        # Track last move
        self.last_move = (self.info['last_row'], self.info['last_action'])

        self.check_game_over(reward, terminated)

        self.ai_thinking = False
        if not self.game_over:
            self.message = "Your Turn"

    def check_game_over(self, reward, terminated):
        if terminated:
            self.game_over = True
            if reward == 1.0:
                winner_id = self.info['last_player']
                # Get winning positions
                last_row = self.info['last_row']
                last_col = self.info['last_action']
                self.winning_positions = self.env.get_winning_positions(last_row, last_col)

                if winner_id == self.human_player:
                    self.message = "YOU WIN!"
                else:
                    self.message = "AI WINS!"
            else:
                self.message = "DRAW!"

    def run(self):
        clock = pygame.time.Clock()
        ai_move_pending = False

        # If AI goes first, trigger it immediately
        if self.info['current_player'] == self.ai_player:
            self.handle_ai_move()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN and not self.ai_thinking:
                    mouse_pos = event.pos

                    # Check button clicks
                    if self.restart_button.collidepoint(mouse_pos):
                        self.restart_game()
                        ai_move_pending = self.info['current_player'] == self.ai_player
                        continue

                    if self.switch_button.collidepoint(mouse_pos):
                        self.toggle_first_player()
                        ai_move_pending = self.info['current_player'] == self.ai_player
                        continue

                    # Handle board clicks (only if game not over)
                    if not self.game_over:
                        pos_x = event.pos[0]
                        pos_y = event.pos[1]
                        board_bottom = OFFSET_Y + BOARD_ROWS * SQUARE_SIZE
                        if OFFSET_X <= pos_x <= OFFSET_X + BOARD_COLS * SQUARE_SIZE and pos_y < board_bottom:
                            col = int((pos_x - OFFSET_X) // SQUARE_SIZE)

                            # Check validity
                            valid_actions = self.obs['action_mask']
                            if valid_actions[col]:
                                self.obs, reward, terminated, _, self.info = self.env.step(col)

                                # Track last move
                                self.last_move = (self.info['last_row'], self.info['last_action'])

                                self.check_game_over(reward, terminated)
                                self.draw_board()  # Force update before AI starts

                                # Trigger AI if game continues
                                if not self.game_over:
                                    # Small delay for visual clarity
                                    pygame.time.wait(500)
                                    self.handle_ai_move()

            # Handle pending AI move after restart/switch
            if ai_move_pending and not self.ai_thinking:
                self.draw_board()
                pygame.time.wait(300)
                self.handle_ai_move()
                ai_move_pending = False

            self.draw_board()
            clock.tick(30)


def get_default_model_path():
    """Get the default model path, checking bundle first, then local."""
    # Check for bundled model first (PyInstaller bundle)
    bundled_path = get_resource_path("best_model.pt")
    if os.path.exists(bundled_path):
        return bundled_path
    # Fall back to checkpoints directory (development)
    return "checkpoints/best_model.pt"


def main():
    parser = argparse.ArgumentParser(description="Play Connect-4 against the AI (GUI)")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--simulations", type=int, default=400,
        help="MCTS simulations per AI move"
    )
    parser.add_argument(
        "--cpuct", type=float, default=2.0,
        help="MCTS exploration parameter"
    )
    parser.add_argument(
        "--ai-first", action="store_true",
        help="Let the AI play first (as Red)"
    )

    args = parser.parse_args()

    model_path = args.model if args.model else get_default_model_path()
    gui = Connect4GUI(model_path, args.simulations, args.cpuct, args.ai_first)
    gui.run()


if __name__ == "__main__":
    main()
