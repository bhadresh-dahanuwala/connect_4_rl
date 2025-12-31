import numpy as np

from connect_4_rl.mcts import MCTS


class AlphaZeroAgent:
    def __init__(self, model, num_simulations=100, c_puct=1.0, device='cpu'):
        self.model = model
        self.mcts = MCTS(model, num_simulations, c_puct, device)

    def select_move(self, env, temperature=1.0, add_noise=False):
        """
        Selects a move using MCTS.
        Returns:
            action (int): The selected column.
            action_probs (np.ndarray): The probability distribution over actions.
        """
        action_probs = self.mcts.get_action_probs(env, temperature, add_noise)
        action = np.random.choice(len(action_probs), p=action_probs)
        return action, action_probs
