import copy
import random
import math
import numpy as np
from game2048.game2048 import Game2048Env
from game2048.agent.expectimax import expectimax
from math import inf

env = Game2048Env()

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

class HEU_Approximator:

    def __init__(self):
        self.weight = np.array(
            [[2**0, 2**1, 2**2, 2**3],
            [2**7, 2**6, 2**5, 2**4],
            [2**8, 2**9, 2**10, 2**11],
            [2**15, 2**14, 2**13, 2**12]],
            dtype=np.float32
        )

    def value(self, state):
        return (state * self.weight).sum()

class HEU_MCTS_Node:
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
class HEU_MCTS:
    def __init__(self, env, iterations=500, exploration_constant=1.41, rollout_depth=0, gamma=0.99):
        self.env = env
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.approximator = HEU_Approximator()
        self.max_value = -inf
        self.min_value = inf

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
            uct = child.total_reward + self.c * math.sqrt(math.log(node.visits) / child.visits)
            if uct > max_uct:
                max_uct = uct
                chd = child
        return chd
    
    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        reward = 0
        for _ in range(depth):
            action = np.random.choice(4, 1)[0]
            state, reward, done, _ = sim_env.step(action)
            if done:
                reward = -2**50
                break
        # print(reward, self.approximator.value(state))
        estimate = reward + self.approximator.value(state)
        estimate = math.log2(estimate) if estimate > 0 else estimate
        self.max_value = max(self.max_value, estimate)
        self.min_value = min(self.min_value, estimate)
        return (estimate - self.min_value) / (self.max_value - self.min_value) if self.max_value - self.min_value != 0 else 0

    # def rollout(self, sim_env, depth):
    #     legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
    #     score, _ = expectimax(self.approximator, sim_env.board, sim_env.score, legal_actions, depth=1, max_depth=depth)
    #     return score

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
            child = HEU_MCTS_Node(state, 0, parent=node, action=action)
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
    
if __name__ == '__main__':
    env = Game2048Env()
    heu_mcts = HEU_MCTS(env, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99)

    state = env.reset()
    env.render()

    done = False
    while not done:
        # Create the root node from the current state
        root = HEU_MCTS_Node(state, env.score)

        # Run multiple simulations to build the MCTS tree
        for _ in range(heu_mcts.iterations):
            heu_mcts.run_simulation(root)

        # Select the best action (based on highest visit count)
        best_act, _ = heu_mcts.best_action_distribution(root)
        print("HEU-MCTS selected action:", best_act, "Score:", env.score)
        print("State:\n", state)

        # Execute the selected action and update the state
        state, reward, done, _ = env.step(best_act)
        env.render(action=best_act)

    print("Game over, final score:", env.score)
