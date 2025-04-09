import numpy as np
import random

import copy
import random
import math
import sys
import numpy as np
import torch

from connect6_value_approximator import Connect6ValueNet

# UCT Node for MCTS
class Connect6_UCTNode:
    def __init__(self, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [(r, c) for r in range(self.state.shape[0]) for c in range(self.state.shape[1]) if self.state[r, c] == 0]

    def fully_expanded(self):
		# A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


class Connect6_UCTMCTS:
    def __init__(self, env, iterations=10, exploration_constant=1.41, rollout_depth=10, device='mps'):
        self.env = env
        self.iterations = iterations
        self.c = exploration_constant  # Balances exploration and exploitation
        self.rollout_depth = rollout_depth
        self.board_size = self.env.board_size
        self.action_space = self.board_size ** 2
        self.value_approximator = Connect6ValueNet().to(device)
        self.device = device

    def create_env_from_state(self, state, score):
        """
        Creates a deep copy of the environment with a given board state and score.
        """
        new_env = copy.deepcopy(self.env)
        new_env.game.board = state.copy()
        new_env.score = score
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

    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout from the current state up to the specified depth.
        # state = sim_env.game.board
        # for _ in range(depth):
        #     action = [random.choice([(r, c) for r in range(state.shape[0]) for c in range(state.shape[1]) if state[r, c] == 0])]
        #     state, reward, done, _ = sim_env.step(action)
        #     if done:
        #         break
        # return reward
        state = torch.from_numpy(sim_env.game.board).long()
        empty = (state == 0).float()
        player = (state == 1).float()
        opponent = (state == 2).float()
        state_input = torch.stack((empty, player, opponent), dim=0).unsqueeze(dim=0).to(self.device)
        return self.value_approximator(state_input).item()

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
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            sim_env = self.create_env_from_state(node.state, node.score)
            state, reward, done, _ = sim_env.step([action])
            # create new node
            child = Connect6_UCTNode(state, 0, parent=node, action=action)
            sim_env = self.create_env_from_state(child.state, child.score)
            node.children.update({action: child})

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
        
        for _ in range(self.iterations):
            self.run_simulation(root)

        best_action, _ = self.best_action_distribution(root)
        return best_action


class Connect6_Wrapper:

    def __init__(self, env):
        super().__init__()
        self.board_size = env.size
        self.game = env
        self.current_player = 1  # 1 for Black, 2 for White
        self.score = 0

    def reset(self, *, seed=None, options=None):
        self.game.reset_board()
        self.current_player = 1
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

        self.current_player = 3 - self.current_player
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return self.game.board.copy()

    def render(self):
        self.game.show_board()