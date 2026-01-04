import os
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from connect_4_rl.agent import AlphaZeroAgent
from connect_4_rl.env import Connect4Env, PLAYER_BLUE, PLAYER_RED
from connect_4_rl.model import Connect4Net


def log(message, section=False):
    """Print a message with timestamp prefix."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if section:
        print(f"\n[{timestamp}] {'─' * 50}")
        print(f"[{timestamp}] {message}")
        print(f"[{timestamp}] {'─' * 50}")
    else:
        print(f"[{timestamp}] {message}")


def self_play_worker(model_config, challenger_state, champion_state, simulations, c_puct, p1_is_challenger):
    """Run one self-play game: challenger vs champion.

    Training examples are collected only from the challenger's moves.
    """
    torch.set_num_threads(1)

    challenger = Connect4Net(
        num_channels=model_config['num_channels'],
        num_res_blocks=model_config['num_blocks']
    ).to('cpu')
    challenger.load_state_dict(challenger_state)
    challenger.eval()

    champion = Connect4Net(
        num_channels=model_config['num_channels'],
        num_res_blocks=model_config['num_blocks']
    ).to('cpu')
    champion.load_state_dict(champion_state)
    champion.eval()

    env = Connect4Env()
    obs, info = env.reset()
    examples = []

    challenger_agent = AlphaZeroAgent(challenger, simulations, c_puct, 'cpu')
    champion_agent = AlphaZeroAgent(champion, simulations, c_puct, 'cpu')

    # Assign agents to players
    if p1_is_challenger:
        agent1, agent2 = challenger_agent, champion_agent
        challenger_player = PLAYER_RED
    else:
        agent1, agent2 = champion_agent, challenger_agent
        challenger_player = PLAYER_BLUE

    step_count = 0
    while True:
        current_player = info['current_player']
        board = obs['observation']
        action_mask = obs['action_mask']

        # Select agent based on current player
        if current_player == PLAYER_RED:
            current_agent = agent1
            is_challenger_turn = p1_is_challenger
        else:
            current_agent = agent2
            is_challenger_turn = not p1_is_challenger

        # Temperature and noise for exploration
        if step_count < 20:
            temp = 1.0
            noise = True
        else:
            temp = 0.0
            noise = False

        action, action_probs = current_agent.select_move(env, temp, add_noise=noise)

        # Only collect examples from challenger's moves
        if is_challenger_turn:
            canonical_board = board.copy()
            if current_player == PLAYER_BLUE:
                canonical_board[0] = board[1]
                canonical_board[1] = board[0]

            # Symmetries
            sym_board = np.flip(canonical_board, axis=2).copy()
            sym_probs = np.flip(action_probs).copy()

            examples.append([canonical_board, action_probs, current_player])
            examples.append([sym_board, sym_probs, current_player])

        obs, reward, terminated, _, info = env.step(action)
        step_count += 1

        if terminated:
            winner = info['last_player']
            result = 1 if reward == 1.0 else 0

            result_examples = []
            for ex_board, ex_probs, ex_player in examples:
                outcome = 0
                if reward == 1.0:
                    outcome = 1 if ex_player == info['last_player'] else -1
                result_examples.append((ex_board, ex_probs, outcome))

            return result_examples, step_count, result, winner


def eval_worker(model_config, challenger_state, champion_state, simulations, c_puct, p1_is_challenger):
    """Run one evaluation game in a worker process."""
    torch.set_num_threads(1)

    challenger = Connect4Net(
        num_channels=model_config['num_channels'],
        num_res_blocks=model_config['num_blocks']
    ).to('cpu')
    challenger.load_state_dict(challenger_state)
    challenger.eval()

    champion = Connect4Net(
        num_channels=model_config['num_channels'],
        num_res_blocks=model_config['num_blocks']
    ).to('cpu')
    champion.load_state_dict(champion_state)
    champion.eval()

    env = Connect4Env()
    _, info = env.reset()

    if p1_is_challenger:
        agent1 = AlphaZeroAgent(challenger, simulations, c_puct, 'cpu')
        agent2 = AlphaZeroAgent(champion, simulations, c_puct, 'cpu')
    else:
        agent1 = AlphaZeroAgent(champion, simulations, c_puct, 'cpu')
        agent2 = AlphaZeroAgent(challenger, simulations, c_puct, 'cpu')

    step_count = 0
    terminated = False
    while not terminated:
        if step_count < 8:
            temp = 0.8
            noise = True
        else:
            temp = 0.0
            noise = False

        current_player = info['current_player']
        if current_player == PLAYER_RED:
            action, _ = agent1.select_move(env, temp, add_noise=noise)
        else:
            action, _ = agent2.select_move(env, temp, add_noise=noise)

        _, reward, terminated, _, info = env.step(action)
        step_count += 1

    if reward == 1.0:
        winner_id = info['last_player']
        if winner_id == PLAYER_RED:
            result = 1 if p1_is_challenger else -1
        else:
            result = -1 if p1_is_challenger else 1
    else:
        result = 0

    return result, step_count


class Trainer:
    def __init__(self, args):
        self.args = args
        self.cpu_device = torch.device('cpu')
        self.train_device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        self.model = Connect4Net(
            num_channels=self.args['num_channels'],
            num_res_blocks=self.args['num_blocks']
        ).to(self.train_device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.args['lr'], weight_decay=1e-4
        )

        # Print Configuration and Model Summary
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log("TRAINING INITIALIZED", section=True)
        log(f"  Model: {self.args['num_blocks']} blocks, {self.args['num_channels']} channels, {num_params:,} params")
        eval_sims = self.args.get('eval_simulations', self.args['num_simulations'])
        log(f"  Training: {self.args['iterations']} iters, {self.args['num_self_play_games']} games/iter, {self.args['num_simulations']} sims (eval: {eval_sims})")
        log(f"  Learning: LR={self.args['lr']}, batch={self.args['batch_size']}, epochs={self.args['epochs']}")
        log(f"  Device: {self.train_device}")

        # Scheduler: Reduce LR when loss plateaus
        # threshold=0.01 means loss must improve by 1% to count as progress
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=4,
            threshold=0.01,
            threshold_mode='rel',
            cooldown=10,
            min_lr=1e-6,
        )

        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.examples = deque(maxlen=self.args['max_buffer_size'])
        self.start_iteration = 0

        if self.args['resume']:
            if os.path.isfile(self.args['resume']):
                checkpoint = torch.load(self.args['resume'], weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_iteration = checkpoint['iteration'] + 1
                log(f"  Resuming from iteration {self.start_iteration}")
            else:
                log(f"  Warning: Checkpoint not found at {self.args['resume']}")

        # Initialize persistent executor
        self.executor = ProcessPoolExecutor(max_workers=self.args['workers'])

        # Track best model state separately for evaluation
        # best_model_state is only set after first training iteration
        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            best_checkpoint = torch.load(best_model_path, weights_only=False, map_location='cpu')
            self.best_model_state = {k: v.cpu() for k, v in best_checkpoint['model_state_dict'].items()}
            log(f"  Loaded best model from {best_model_path}")
        else:
            # Don't save yet - will save after first training iteration
            self.best_model_state = None


    def run(self):
        # Track if next iteration should reset to Best (for iteration 2)
        self.reset_to_best = False

        try:
            for i in range(self.start_iteration, self.args['iterations']):
                log(f"ITERATION {i+1}/{self.args['iterations']}", section=True)

                # 1. Self Play
                if self.best_model_state is None:
                    # Iteration 1: M0 vs M0
                    m0_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                    current_model_state = m0_state
                    new_examples = self.self_play(m0_state, m0_state)
                elif self.reset_to_best:
                    # Iteration 2: Reset to Best, Best vs Best
                    self.model.load_state_dict(self.best_model_state)
                    current_model_state = self.best_model_state
                    new_examples = self.self_play(self.best_model_state, self.best_model_state)
                    self.reset_to_best = False
                else:
                    # Iteration 3+: M{prev_iter} vs Best
                    # challenger = current model (M{prev_iter}, being trained)
                    # champion = best model (to beat)
                    # Examples are collected from challenger's moves
                    current_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                    new_examples = self.self_play(current_model_state, self.best_model_state)

                self.examples.extend(new_examples)

                # 2. Train - creates new model from Best (or M0 for first iteration)
                train_loss = self.train(list(self.examples))

                # Step the scheduler after a burn-in period (skip first 20 iterations)
                old_lr = self.optimizer.param_groups[0]['lr']
                if i >= 20:
                    self.scheduler.step(train_loss)
                current_lr = self.optimizer.param_groups[0]['lr']

                if current_lr < old_lr:
                    log(f"[LR] Reduced: {old_lr:.6f} -> {current_lr:.6f}")

                # 3. Get trained model state (M1, M2, etc.)
                trained_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

                # Save checkpoint state
                checkpoint_state = {
                    'iteration': i,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()
                }

                torch.save(
                    checkpoint_state,
                    os.path.join(self.checkpoint_dir, f"checkpoint_{i}.pt")
                )

                # Clean up old checkpoints (keep only last 5)
                self._cleanup_old_checkpoints(keep_last=5)

                # 4. Evaluate and update Best
                if self.best_model_state is None:
                    # First iteration: M0 vs M1, pick the better one
                    win_ratio = self.evaluate(trained_state, m0_state)
                    log(f"[Summary] Loss={train_loss:.4f}, LR={current_lr:.6f}, Buffer={len(self.examples)}, Status=INITIAL")
                    if win_ratio >= 0.5:
                        log(f"  > Trained model (M1) wins {win_ratio:.0%}. M1 becomes Best.")
                        self.best_model_state = trained_state
                    else:
                        log(f"  > Random model (M0) wins {1-win_ratio:.0%}. M0 becomes Best.")
                        self.best_model_state = m0_state
                    torch.save(
                        {'model_state_dict': self.best_model_state},
                        os.path.join(self.checkpoint_dir, "best_model.pt")
                    )
                    # Next iteration (iter 2) should reset to Best
                    self.reset_to_best = True
                    continue

                win_ratio = self.evaluate(trained_state, self.best_model_state)

                # Summary
                status = "IMPROVED" if win_ratio >= 0.62 else "REGRESSED" if win_ratio <= 0.45 else "STABLE"
                log(f"[Summary] Loss={train_loss:.4f}, LR={current_lr:.6f}, Buffer={len(self.examples)}, Status={status}")

                if win_ratio >= 0.62:
                    log(f"  > New Champion! Saving best_model.pt (Win Rate: {win_ratio:.0%})")
                    self.best_model_state = trained_state
                    torch.save(
                        {'model_state_dict': self.best_model_state},
                        os.path.join(self.checkpoint_dir, "best_model.pt")
                    )
                else:
                    log(f"  > Model did not improve (Win Rate: {win_ratio:.0%}). Keeping previous best.")

        finally:
            self.executor.shutdown()

    def _cleanup_old_checkpoints(self, keep_last=5):
        """Remove old checkpoints, keeping only the most recent ones."""
        import glob
        import re

        checkpoint_pattern = os.path.join(self.checkpoint_dir, "checkpoint_*.pt")
        checkpoints = glob.glob(checkpoint_pattern)

        # Extract iteration numbers and sort
        def get_iteration(path):
            match = re.search(r'checkpoint_(\d+)\.pt$', path)
            return int(match.group(1)) if match else -1

        checkpoints_with_iter = [(cp, get_iteration(cp)) for cp in checkpoints]
        checkpoints_with_iter.sort(key=lambda x: x[1], reverse=True)

        # Remove old checkpoints (keep best_model.pt always)
        for checkpoint_path, _ in checkpoints_with_iter[keep_last:]:
            try:
                os.remove(checkpoint_path)
            except OSError:
                pass

    def self_play(self, challenger_state, champion_state):
        """Run self-play games: challenger vs champion."""
        examples = []
        num_games = self.args['num_self_play_games']

        model_config = {
            'num_channels': self.args['num_channels'],
            'num_blocks': self.args['num_blocks']
        }

        log(f"[Self-Play] Running {num_games} games ({self.args['workers']} workers)...")

        futures = [
            self.executor.submit(
                self_play_worker, model_config, challenger_state, champion_state,
                self.args['num_simulations'], self.args['c_puct'], i % 2 == 0
            )
            for i in range(num_games)
        ]

        wins = {1: 0, 2: 0, 'draw': 0}
        total_steps = 0
        for idx, future in enumerate(as_completed(futures)):
            game_examples, step_count, result, winner = future.result()
            examples += game_examples
            total_steps += step_count
            if result == 1:
                wins[winner] += 1
            else:
                wins['draw'] += 1

            # Log progress every 10 games
            if (idx + 1) % 10 == 0:
                log(f"[Self-Play] Progress: {idx + 1}/{num_games} games finished")

        avg_steps = total_steps / num_games
        log(f"[Self-Play] Done: {len(examples)} examples, avg {avg_steps:.1f} steps/game")
        log(f"[Self-Play] Results: P1={wins[1]}, P2={wins[2]}, Draws={wins['draw']}")
        return examples

    def train(self, examples):
        log(f"[Training] {len(examples)} examples, {self.args['epochs']} epochs...")
        self.model.train()
        batch_size = self.args['batch_size']
        epochs = self.args['epochs']

        first_loss = None
        for epoch in range(epochs):
            random.shuffle(examples)
            batch_count = int(len(examples) / batch_size)
            if batch_count == 0:
                continue

            total_loss = 0
            total_loss_v = 0
            total_loss_pi = 0

            for i in range(batch_count):
                sample_idx = range(i * batch_size, (i + 1) * batch_size)
                batch = [examples[k] for k in sample_idx]

                boards, pis, vs = zip(*batch)

                boards = torch.tensor(
                    np.array(boards), dtype=torch.float32
                ).to(self.train_device)
                target_pis = torch.tensor(
                    np.array(pis), dtype=torch.float32
                ).to(self.train_device)
                target_vs = torch.tensor(
                    np.array(vs), dtype=torch.float32
                ).view(-1, 1).to(self.train_device)

                out_pi, out_v = self.model(boards)

                loss_v = F.mse_loss(out_v, target_vs)
                loss_pi = -(target_pis * F.log_softmax(out_pi, dim=1)).sum(dim=1).mean()

                loss = loss_v + loss_pi

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_loss_v += loss_v.item()
                total_loss_pi += loss_pi.item()

            avg_loss = total_loss / batch_count
            avg_loss_v = total_loss_v / batch_count
            avg_loss_pi = total_loss_pi / batch_count

            if first_loss is None:
                first_loss = avg_loss

        log(f"[Training] Loss: {first_loss:.4f} -> {avg_loss:.4f} (V={avg_loss_v:.4f}, P={avg_loss_pi:.4f})")
        return avg_loss

    def evaluate(self, challenger_state, champion_state):
        challenger_wins = 0
        champion_wins = 0
        draws = 0

        games = self.args['num_eval_games']
        if games == 0:
            return 0

        model_config = {
            'num_channels': self.args['num_channels'],
            'num_blocks': self.args['num_blocks']
        }

        log(f"[Eval] Running {games} games...")

        eval_sims = self.args.get('eval_simulations', self.args['num_simulations'])
        futures = [
            self.executor.submit(
                eval_worker, model_config, challenger_state, champion_state,
                eval_sims, self.args['c_puct'], i % 2 == 0
            )
            for i in range(games)
        ]

        total_steps = 0
        for future in as_completed(futures):
            result, step_count = future.result()
            total_steps += step_count
            if result == 1:
                challenger_wins += 1
            elif result == -1:
                champion_wins += 1
            else:
                draws += 1

        win_ratio = (challenger_wins + 0.5 * draws) / games
        avg_steps = total_steps / games
        log(f"[Eval] New={challenger_wins}, Old={champion_wins}, Draws={draws} | Win Rate: {win_ratio:.0%} | Avg Steps: {avg_steps:.1f}")
        return win_ratio