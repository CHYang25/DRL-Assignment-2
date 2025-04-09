import copy
import math
import random
import numpy as np
from collections import defaultdict
from ntuple_td_learning import rotation_90, transpose, NTupleApproximator
from game2048.game2048 import Game2048Env
from td_mcts import TD_MCTS, TD_MCTS_Node
import pickle

# TODO: Define the action transformation functions (i.e., rot90_action, rot180_action, etc.)
# Note: You have already defined transformation functions for patterns before.
def orig_action(x):
    return x

def rot90_action(x):
    return [3, 2, 0, 1][x]

def rot180_action(x):
    return [1, 0, 3, 2][x]

def rot270_action(x):
    return [2, 3, 1, 0][x]

def rot_trans_action(x):
    return [2, 3, 0, 1][x]

def rot90_trans_action(x):
    return [0, 1, 3, 2][x]

def rot180_trans_action(x):
    return [3, 2, 1, 0][x]

def rot270_trans_action(x):
    return [1, 0, 2, 3][x]

action_transformations = [orig_action, rot90_action, rot180_action, rot270_action, rot_trans_action, rot90_trans_action, rot180_trans_action, rot270_trans_action]

def softmax(x):
    x_exp = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return x_exp / np.sum(x_exp)

def create_default_float_dict():
    return defaultdict(float)

# Note: PolicyApproximator is similar to the value approximator but differs in key aspects.
class PolicyApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want.
        """
        self.board_size = board_size
        self.patterns = patterns
        self.actions = [0, 1, 2, 3]
        # Weight structure: [pattern_idx][feature_key][action]
        self.weights = [defaultdict(create_default_float_dict) for _ in range(len(patterns))]
        # Generate the 8 symmetrical transformations for each pattern and store their types.
        self.symmetry_patterns = []
        self.symmetry_types = []  # Store the type of symmetry transformation (rotation or reflection)
        for pattern in self.patterns:
            syms, types = self.generate_symmetries(pattern)
            self.symmetry_patterns.append(syms)
            self.symmetry_types.append(types)

        # TODO: Define corresponding action transformation functions for each symmetry.
        self.action_trans_syms = []
        for stypes in self.symmetry_types:
            action_trans = [action_transformations[stype] for stype in stypes]
            self.action_trans_syms.append(action_trans)

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        syms = []
        for _ in range(4):
            syms.append(pattern)
            pattern = rotation_90(pattern)
        pattern = transpose(pattern)
        syms = []
        for _ in range(4):
            syms.append(pattern)
            pattern = rotation_90(pattern)
        return syms, list(range(8))


    def tile_to_index(self, tile):
        return 0 if tile == 0 else int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[x, y]) for x, y in coords)

    def predict(self, board):
        # TODO: Predict the policy (probability distribution over actions) given the board state.
        logits = np.zeros(4)
        for idx, symmetries in enumerate(self.symmetry_patterns):
            for pattern in symmetries:
                feature = self.get_feature(board, pattern)
                logits += np.array([self.weights[idx][feature][a] for a in self.actions])

        distribution = softmax(logits)
        return np.random.choice(4, 1, p=distribution)[0]


    def update(self, board, target_distribution, alpha=0.1):
        # TODO: Update policy based on the target distribution.
        logits = np.zeros(4)
        feature_indices = []
        for idx, symmetries in enumerate(self.symmetry_patterns):
            for pattern, action_trans in zip(symmetries, self.action_trans_syms[idx]):
                feature = self.get_feature(board, pattern)
                feature_indices.append((idx, feature, action_trans))
                logits += np.array([self.weights[idx][feature][a] for a in self.actions])

        distribution = softmax(logits)
        gradient = target_distribution - distribution
        for idx, feature, action_trans in feature_indices:
            for a in self.actions:
                transformed_action = action_trans(a)
                self.weights[idx][feature][transformed_action] += alpha * gradient[a]

def self_play_training_policy_with_td_mcts(env, td_mcts, policy_approximator, num_episodes=50):
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            # Create the root node for the TD-MCTS tree
            root = TD_MCTS_Node(state, env.score)

            # Run multiple simulations to build the MCTS search tree
            for _ in range(td_mcts.iterations):
                td_mcts.run_simulation(root)

            best_action, target_distribution = td_mcts.best_action_distribution(root)

            # TODO: Update the NTuple Policy Approximator using the MCTS action distribution
            # Here, we use the MCTS result directly as the label to update the policy
            policy_approximator.update(state, target_distribution)


            # Execute the selected action in the real environment
            state, reward, done, _ = env.step(best_action)

        print(f"Episode {episode+1}/{num_episodes} finished, final score: {env.score}")

if __name__ == '__main__':
    env = Game2048Env()

    # TODO: Define your own pattern
    patterns = [
        np.argwhere(np.array(
            [[0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],]
        )),
        np.argwhere(np.array(
            [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],]
        )),
        np.argwhere(np.array(
            [[1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],]
        )),
        np.argwhere(np.array(
            [[0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],]
        )),
    ]

    approximator = pickle.load(open('./n-tuple-approximator.pkl', 'rb'))

    policy_approximator = PolicyApproximator(board_size=4, patterns=patterns)
    td_mcts = TD_MCTS(env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=10, gamma=0.99)

    self_play_training_policy_with_td_mcts(env, td_mcts, policy_approximator, num_episodes=50)

    pickle.dump(policy_approximator, open("./policy_approximator.pkl", "wb"))