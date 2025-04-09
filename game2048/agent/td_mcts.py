import copy
import random
import math
import numpy as np
from game2048.game2048 import Game2048Env

env = Game2048Env()

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
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
        # List of untried actions based on the current state's legal moves
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        # self.value_avg = sum([sum(w.values()) for w in self.approximator.weights]) / sum([len(w) for w in approximator.weights])
        # print(self.value_avg)
        # self.value_norm = 20000

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
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
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        for _ in range(depth):
            action = np.random.choice(4, 1)[0]
            state, reward, done, _ = sim_env.step(action)
            if done:
                break
        # print(reward, self.approximator.value(state))
        estimate = self.approximator.value(state) / 2000
        
        return reward + estimate

    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
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

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        while node.fully_expanded():
            child = self.select_child(node)
            if child is None:
                break
            node = child

        # TODO: Expansion: If the node is not terminal, expand an untried action.
        if not node.fully_expanded(): # this suggests that node is not a terminal node, and the node still could be expanded
            action = random.choice(node.untried_actions)

            node.untried_actions.remove(action)
            sim_env = self.create_env_from_state(node.state, node.score)
            state, reward, done, _ = sim_env.step(action)
            # create a new node
            child = TD_MCTS_Node(state, 0, parent=node, action=action)
            sim_env = self.create_env_from_state(child.state, child.score)
            node.children.update({action: child})

            node = child

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution
