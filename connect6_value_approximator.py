import copy
import random
import math
import numpy as np
from collections import defaultdict
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class Connect6ValueNet(nn.Module):
    def __init__(self, board_size=19):
        super().__init__()
        self.board_size = board_size
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.head(x)
        return x


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
