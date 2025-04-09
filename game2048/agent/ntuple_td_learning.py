from game2048.game2048 import Game2048Env
import copy
import random
import math
import numpy as np
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt


# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------
def rotation_90(pattern):
    rot_pat = []
    for x, y in pattern:
        rot_pat.append([y, 4 - 1 - x])
    return np.array(rot_pat)

def transpose(pattern):
    trans_pat = []
    for x, y in pattern:
        trans_pat.append([y, x])
    return np.array(trans_pat)

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        for i, pattern in enumerate(self.patterns):
            syms = self.generate_symmetries(pattern, i)
            self.symmetry_patterns.append(syms)
        self.num_syms = sum([len(s) for s in self.symmetry_patterns])
        # print([len(s) for s in self.symmetry_patterns])
        # print(self.symmetry_patterns)

    def generate_symmetries(self, pattern, idx):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        syms = []
        for _ in range(4):
            syms.append(pattern)
            pattern = rotation_90(pattern)
        pattern = transpose(pattern)
        syms2 = []
        for _ in range(4):
            syms2.append(pattern)
            pattern = rotation_90(pattern)
        return syms + syms2

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        return int(math.log(tile, 2)) if tile > 0 else 0

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[x, y]) for x, y in coords)

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        total_value = 0
        for idx, symmetries in enumerate(self.symmetry_patterns):
            for pattern in symmetries:
                feature = self.get_feature(board, pattern)
                total_value += self.weights[idx][feature]
        return total_value

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        for idx, symmetries in enumerate(self.symmetry_patterns):
            for pattern in symmetries:
                feature = self.get_feature(board, pattern)
                self.weights[idx][feature] += alpha * delta / self.num_syms

def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
    """
    final_scores = []
    success_flags = []

    for episode in range(num_episodes):
        state = env.reset()
        trajectory = []  # Store trajectory data if needed
        previous_score = 0
        done = False
        max_tile = np.max(state)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            # TODO: action selection
            # Note: TD learning works fine on 2048 without explicit exploration, but you can still try some exploration methods.
            # if np.random.uniform(0, 1) > epsilon:
            values = []
            for a in legal_moves:
                tmp_env = copy.deepcopy(env)
                next_state, new_score, _, _ = tmp_env.step(a)
                reward = new_score - previous_score
                values.append(reward + gamma*approximator.value(next_state))

            action = legal_moves[np.argmax(values)]
                # values = np.array(values)
                # action = legal_moves[np.random.choice(np.where(values == values.max())[0])]
            # else:
            #     action = np.random.choice(4, 1)[0]

            next_state, new_score, done, _ = env.step(action)
            incremental_reward = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))

            # TODO: Store trajectory or just update depending on the implementation
            td_error = incremental_reward + gamma * approximator.value(next_state) * int(not done) - approximator.value(state)
            approximator.update(state, td_error, alpha)

            state = next_state.copy()

        # TODO: If you are storing the trajectory, consider updating it now depending on your implementation.

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(sum([len(s) for s in approximator.weights]), sum([12**len(x) for x in approximator.symmetry_patterns]))
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")

    return final_scores

if __name__ == '__main__':
    # TODO: Define your own n-tuple patterns
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
        # np.argwhere(np.array(
        #     [[0, 0, 0, 0],
        #      [0, 1, 1, 0],
        #      [0, 1, 1, 0],
        #      [0, 0, 0, 0],]
        # )),
    ]

    approximator = NTupleApproximator(board_size=4, patterns=patterns)

    env = Game2048Env()

    # Run TD-Learning training
    # Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
    # However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.
    final_scores = td_learning(env, approximator, num_episodes=100000, alpha=0.1, gamma=0.99, epsilon=0.1)
    plt.plot(final_scores)
    plt.xlabel("Episodes")
    plt.ylabel("Average Score")
    plt.title("N-tuple approximator")
    plt.savefig('./n_tuple_td_learning.png')

    pickle.dump(approximator, open("./n-tuple-approximator.pkl", 'wb'))