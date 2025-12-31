import copy
import math

import numpy as np
import torch

from connect_4_rl.env import PLAYER_BLUE


class MCTS:
    def __init__(self, model, num_simulations=100, c_puct=1.0, device='cpu'):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device

    def search(self, env, add_noise=False):
        root = Node(0)

        # Expand root
        obs = env.get_observation()
        action_mask = obs['action_mask']

        self._expand_node(root, obs, action_mask, env.current_player)

        if add_noise:
            # Add Dirichlet noise to root children
            epsilon = 0.25
            alpha = 1.0  # Suitable for small action spaces like Connect 4

            valid_actions = list(root.children.keys())
            if valid_actions:
                noise = np.random.dirichlet([alpha] * len(valid_actions))

                for i, action in enumerate(valid_actions):
                    child = root.children[action]
                    child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

        for _ in range(self.num_simulations):
            node = root
            path = [node]
            simulation_env = copy.deepcopy(env)
            terminated = False
            reward = 0.0

            # Selection
            while node.is_expanded():
                action, node = node.select_child(self.c_puct)
                _, reward, terminated, _, _ = simulation_env.step(action)
                path.append(node)

                if terminated:
                    break

            # Evaluation
            leaf_value = 0.0
            if terminated:
                # If reward is 1.0 (Win), it means the player who just moved (Parent of current node) won.
                # So the current node (next player) is in a losing state -> -1.0
                # If reward is 0.0 (Draw), it's 0.0
                if reward == 1.0:
                    leaf_value = -1.0
                else:
                    leaf_value = 0.0
            else:
                # Expansion
                obs = simulation_env.get_observation()
                action_mask = obs['action_mask']
                leaf_value = self._expand_node(node, obs, action_mask, simulation_env.current_player)

            # Backpropagation
            current_value = leaf_value
            for node in reversed(path):
                node.visit_count += 1
                node.value_sum += current_value
                current_value = -current_value

        return root

    def _expand_node(self, node, obs, action_mask, current_player):
        board = obs['observation']
        canonical_board = board.copy()

        # Canonicalize board so Plane 0 is always current player
        if current_player == PLAYER_BLUE:
            canonical_board[0] = board[1]
            canonical_board[1] = board[0]

        state_tensor = torch.tensor(canonical_board, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)

        policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        value = value.item()

        # Masking
        policy_probs = policy_probs * action_mask
        policy_sum = np.sum(policy_probs)

        if policy_sum > 0:
            policy_probs /= policy_sum
        else:
            policy_probs = action_mask / np.sum(action_mask)

        node.expand(policy_probs)
        return value

    def get_action_probs(self, env, temperature=1.0, add_noise=False):
        root = self.search(env, add_noise=add_noise)

        counts = np.zeros(8)
        for action, child in root.children.items():
            counts[action] = child.visit_count

        if temperature == 0:
            action = np.argmax(counts)
            probs = np.zeros(8)
            probs[action] = 1.0
            return probs

        # Avoid division by zero if temperature is very small but not exactly 0
        if temperature < 1e-3:
            action = np.argmax(counts)
            probs = np.zeros(8)
            probs[action] = 1.0
            return probs

        counts = counts ** (1.0 / temperature)
        count_sum = np.sum(counts)
        if count_sum == 0:
            # Should not happen if simulations > 0
            probs = np.ones(8) / 8
        else:
            probs = counts / count_sum

        return probs


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.prior = prior

    def is_expanded(self):
        return len(self.children) > 0

    def select_child(self, c_puct):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            # AlphaZero PUCT formula
            u = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)

            # Q value: value_sum / visit_count
            # If visit_count is 0, Q is 0 (or we could use a default/parent value)
            if child.visit_count > 0:
                q = child.value_sum / child.visit_count
            else:
                q = 0

            score = q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, policy_probs):
        for action, prob in enumerate(policy_probs):
            if prob > 0:
                self.children[action] = Node(prob)
