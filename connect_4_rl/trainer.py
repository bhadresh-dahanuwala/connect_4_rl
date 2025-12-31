import copy
import os
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from connect_4_rl.agent import AlphaZeroAgent
from connect_4_rl.env import Connect4Env, PLAYER_BLUE, PLAYER_RED
from connect_4_rl.model import Connect4Net


def self_play_worker(model_state, simulations, c_puct):
    torch.set_num_threads(1)
    model = Connect4Net().to('cpu')
    model.load_state_dict(model_state)
    model.eval()

    env = Connect4Env()
    obs, info = env.reset()
    examples = []

    agent = AlphaZeroAgent(model, simulations, c_puct, 'cpu')

    step_count = 0
    while True:
        temp = 1.0 if step_count < 15 else 0.0

        action, action_probs = agent.select_move(env, temp, add_noise=True)

        current_player = info['current_player']
        board = obs['observation']

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
            # Determine Result
            winner = info['last_player']
            if reward == 1.0:
                result_str = f"Player {winner} Won"
            else:
                result_str = "Draw"

            result_examples = []
            for ex_board, ex_probs, ex_player in examples:
                outcome = 0
                if reward == 1.0:  # Win
                    outcome = 1 if ex_player == info['last_player'] else -1
                elif reward == 0.0:  # Draw
                    outcome = 0

                result_examples.append((ex_board, ex_probs, outcome))
            return result_examples, step_count, result_str


def eval_worker(challenger_state, champion_state, simulations, c_puct, p1_is_challenger):
    torch.set_num_threads(1)
    challenger = Connect4Net().to('cpu')
    challenger.load_state_dict(challenger_state)
    challenger.eval()

    champion = Connect4Net().to('cpu')
    champion.load_state_dict(champion_state)
    champion.eval()

    env = Connect4Env()
    _, info = env.reset()
    temp = 0.0

    if p1_is_challenger:
        p1_model = challenger
        p2_model = champion
    else:
        p1_model = champion
        p2_model = challenger

    agent1 = AlphaZeroAgent(p1_model, simulations, c_puct, 'cpu')
    agent2 = AlphaZeroAgent(p2_model, simulations, c_puct, 'cpu')

    step_count = 0
    terminated = False
    while not terminated:
        current_player = info['current_player']
        if current_player == PLAYER_RED:
            action, _ = agent1.select_move(env, temp, add_noise=False)
        else:
            action, _ = agent2.select_move(env, temp, add_noise=False)

        _, reward, terminated, _, info = env.step(action)
        step_count += 1

    if reward == 1.0:
        winner_id = info['last_player']
        if winner_id == PLAYER_RED:
            result = 1 if p1_is_challenger else -1
            winner_name = "Challenger" if p1_is_challenger else "Champion"
        else:
            result = -1 if p1_is_challenger else 1
            winner_name = "Champion" if p1_is_challenger else "Challenger"
    else:
        result = 0
        winner_name = "Draw"

    return result, step_count, winner_name


class Trainer:
    def __init__(self, args):
        self.args = args
        self.cpu_device = torch.device('cpu')
        self.train_device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Training device: {self.train_device}")
        print(f"Self-play/Eval device: {self.cpu_device}")

        self.model = Connect4Net().to(self.train_device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)

        # Scheduler: Drop LR by 10x at 30% and 60% of total iterations
        milestones = [int(self.args['iterations'] * 0.3), int(self.args['iterations'] * 0.6)]
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=0.1
        )

        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.examples = deque(maxlen=self.args['max_buffer_size'])
        self.start_iteration = 0

        if self.args['resume']:
            if os.path.isfile(self.args['resume']):
                print(f"Loading checkpoint from {self.args['resume']}")
                checkpoint = torch.load(self.args['resume'])
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.start_iteration = checkpoint['iteration'] + 1
                print(f"Resuming from iteration {self.start_iteration}")
            else:
                print(f"Checkpoint not found at {self.args['resume']}")

        # Initialize persistent executor
        self.executor = ProcessPoolExecutor(max_workers=self.args['workers'])

    def run(self):
        try:
            for i in range(self.start_iteration, self.args['iterations']):
                print(f"Iteration {i+1}/{self.args['iterations']}")

                # 1. Self Play
                # Get CPU state dict for workers
                model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                new_examples = self.self_play(model_state)
                self.examples.extend(new_examples)
                print(f"Replay Buffer Size: {len(self.examples)}")

                # 2. Train
                champion_state = copy.deepcopy(model_state)

                self.train(list(self.examples))

                # Step the scheduler
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Learning Rate: {current_lr}")

                # 3. Evaluate
                print("Evaluating against previous version...")
                challenger_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                win_ratio = self.evaluate(challenger_state, champion_state)
                print(f"Challenger Win Ratio: {win_ratio:.2f}")

                # Save checkpoint state
                checkpoint_state = {
                    'iteration': i,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()
                }

                if win_ratio >= 0.55:
                    print("New Champion!")
                    torch.save(
                        checkpoint_state,
                        os.path.join(self.checkpoint_dir, "best_model.pt")
                    )
                else:
                    print("Challenger failed. Reverting to previous champion.")
                    # Load back original weights to GPU model
                    self.model.load_state_dict(champion_state)

                torch.save(
                    checkpoint_state,
                    os.path.join(self.checkpoint_dir, f"checkpoint_{i}.pt")
                )
        finally:
            self.executor.shutdown()

    def self_play(self, model_state):
        examples = []
        num_games = self.args['num_self_play_games']

        futures = [
            self.executor.submit(
                self_play_worker, model_state,
                self.args['num_simulations'], self.args['c_puct']
            )
            for i in range(num_games)
        ]

        for i, future in enumerate(as_completed(futures)):
            game_examples, step_count, result_str = future.result()
            examples += game_examples
            print(f"[Self Play] Game {i+1} finished in {step_count} steps. Result: {result_str}")

        print(f"Self Play collected {len(examples)} examples")
        return examples

    def train(self, examples):
        print("Training...")
        self.model.train()
        batch_size = self.args['batch_size']
        epochs = self.args['epochs']

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
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Loss: {avg_loss:.4f} (V: {avg_loss_v:.4f}, P: {avg_loss_pi:.4f})"
            )

    def evaluate(self, challenger_state, champion_state):
        challenger_wins = 0
        champion_wins = 0
        draws = 0

        games = self.args['num_eval_games']
        if games == 0:
            return 0

        futures = []
        for i in range(games):
            p1_is_challenger = i % 2 == 0
            futures.append(self.executor.submit(
                eval_worker, challenger_state, champion_state,
                self.args['num_simulations'], self.args['c_puct'], p1_is_challenger
            ))

        for i, future in enumerate(as_completed(futures)):
            result, step_count, winner_name = future.result()
            if result == 1:
                challenger_wins += 1
            elif result == -1:
                champion_wins += 1
            else:
                draws += 1
            print(f"[Evaluation] Game {i+1} finished in {step_count} steps. Winner: {winner_name}")

        print(f"Challenger: {challenger_wins}, Champion: {champion_wins}, Draws: {draws}")
        return (challenger_wins + 0.5 * draws) / games
