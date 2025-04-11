import numpy as np
import random

import copy
import random
import math
import sys
import numpy as np
import torch

from connect6_value_approximator import Connect6ValueApproximator

# UCT Node for MCTS
class Connect6_UCTNode:
    def __init__(self, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state.copy()
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [(r, c) for r in range(self.state.shape[0]) for c in range(self.state.shape[1]) if self.state[r, c] == 0]
        self.locally_fully_expaned = False

    def fully_expanded(self):
		# A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0 or self.locally_fully_expaned
    

class Connect6_UCTMCTS:
    def __init__(self, env, iterations=50, exploration_constant=1.41, rollout_depth=10, local_region_size=8, device='cuda'):
        self.env = env
        self.iterations = iterations
        self.c = exploration_constant  # Balances exploration and exploitation
        self.rollout_depth = rollout_depth
        self.board_size = self.env.board_size
        self.action_space = self.board_size ** 2
        self.value_approximator = Connect6ValueApproximator()
        self.device = device
        self.local_region_size = local_region_size
        self.first_round = True # In the first round, the agent should first predict from placed = 1. BUT for the rest, the agent should predict from placed = 0.

    def create_env_from_state(self, state, score):
        """
        Creates a deep copy of the environment with a given board state and score.
        """
        new_env = copy.deepcopy(self.env)
        new_env.game.board = state.copy()
        new_env.score = score
        if not self.first_round:
            new_env.placed = 0
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent_visits)/child_visits) to select the child
        max_uct = -1
        chd = None
        for act, child in node.children.items():
            # print(node.visits, child.visits)
            uct = child.total_reward + self.c * math.sqrt(math.log(node.visits) / child.visits)
            if uct > max_uct:
                max_uct = uct
                chd = child
        return chd

    def random_sample_action(self, state, untried_action = None):
        key_pos = np.argwhere(state != 0)
        if key_pos.shape[0] == 0:
            min_r = min_c = self.board_size // 2 - self.local_region_size // 2
            max_r = max_c = self.board_size // 2 + self.local_region_size // 2
        else:
            min_r, min_c = key_pos[:, 0].min(), key_pos[:, 1].min()
            max_r, max_c = key_pos[:, 0].max() + 1, key_pos[:, 1].max() + 1
          
        if max_r - min_r < self.local_region_size:
            expand = (self.local_region_size - (max_r - min_r)) // 2
            max_r = np.clip(max_r + expand, a_min=0, a_max=self.board_size)
            min_r = np.clip(min_r - expand, a_min=0, a_max=self.board_size)
        
        if max_c - min_c < self.local_region_size:
            expand = (self.local_region_size - (max_c - min_c)) // 2
            max_c = np.clip(max_c + expand, a_min=0, a_max=self.board_size)
            min_c = np.clip(min_c - expand, a_min=0, a_max=self.board_size)
        
        if untried_action is not None:
            untried_action = np.array(untried_action)
            mask = (untried_action[:, 0] >= min_r) & (untried_action[:, 0] < max_r) & \
                        (untried_action[:, 1] >= min_c) & (untried_action[:, 1] < max_c)
            return [tuple(random.choice(untried_action[mask]))], untried_action[mask].shape[0] == 1
        else:
            return [random.choice([(r, c) for r in range(min_r, max_r) for c in range(min_c, max_c) if state[r, c] == 0])]

    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout from the current state up to the specified depth.
        state = sim_env.game.board
        acc_reward = 0
        for _ in range(depth):
            action = self.random_sample_action(state)
            state, reward, done, _ = sim_env.step(action)
            acc_reward += reward
            if done:
                break

        return acc_reward + self.value_approximator.value(state)

    def backpropagate(self, node, reward):
        # TODO: Propagate the reward up the tree, updating visit counts and total rewards.
        while node.parent:
            node.visits += 1
            node.total_reward += (reward - node.total_reward) / node.visits
            node = node.parent

        # the root needs update as well
        node.visits += 1
        node.total_reward += (reward - node.total_reward) / node.visits

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection: Traverse the tree until reaching a non-fully expanded node.
        while node.fully_expanded():
            child = self.select_child(node)
            if child is None:
                break
            node = child

        # TODO: Expansion: if the node has untried actions, expand one.
        if not node.fully_expanded(): # this suggests that node is not a terminal node, and the node still could be expanded
            action, locally_fully_expanded = self.random_sample_action(node.state, node.untried_actions)
            node.untried_actions.remove(action[0])
            node.locally_fully_expaned = locally_fully_expanded
            sim_env = self.create_env_from_state(node.state, node.score)
            state, reward, done, _ = sim_env.step(action)
            # create new node
            child = Connect6_UCTNode(state, 0, parent=node, action=action[0])
            sim_env = self.create_env_from_state(child.state, child.score)
            node.children.update({action[0]: child})

            node = child

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagation: Update the tree with the rollout reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        '''
        Computes the visit count distribution for each action at the root node.
        '''
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(self.action_space)
        best_visits = -1
        best_action = None
        # print(root.children)
        for action, child in root.children.items():
            flat_action = action[0] * self.board_size + action[1]
            distribution[flat_action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution
    
    def predict_action(self, state):

        root = Connect6_UCTNode(state, 0)
        print(state, file=sys.stderr)
        print(self.value_approximator.value(state), file=sys.stderr)
        
        for _ in range(self.iterations):
            self.run_simulation(root)

        best_action, _ = self.best_action_distribution(root)
        self.first_round = False
        return best_action


class Connect6_Wrapper:

    def __init__(self, env):
        super().__init__()
        self.board_size = env.size
        self.game = env
        self.current_player = 1  # 1 for Black, 2 for White
        self.placed = 1
        self.score = 0

    def reset(self, *, seed=None, options=None):
        self.current_player = 1
        self.placed = 1
        self.score = 0
        return self._get_obs(), {}

    def step(self, action):
        """
        Action: tuple of two flattened indices (0 to size*size - 1)
        """
        move_str = ','.join(f"{self.game.index_to_label(c)}{r+1}" for r, c in action)
        color = 'B' if self.current_player == 1 else 'W'
        self.game.play_move(color, move_str)

        winner = self.game.check_win()
        done = winner != 0 or np.all(self.game.board != 0)
        reward = 0
        if done:
            if winner == self.current_player:
                reward = 1
            elif winner == 3 - self.current_player:
                reward = -1
            else:
                reward = 0  # draw

        self.placed = 1 - self.placed
        self.current_player = 3 - self.current_player + self.placed
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return self.game.board.copy()

    def render(self):
        self.game.show_board()