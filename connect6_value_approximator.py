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

WIN_WEIGHT = 1
OFF_5_WEIGHT = 0.12
OFF_4_WEIGHT = 0.12
OFF_3_WEIGHT = 0.01

LOSE_WEIGHT = -1
DEF_5_WEIGHT = -1
DEF_4_WEIGHT = -1
DEF_3_WEIGHT = -0.01

def pattern_generator():
    ## Offensive Pattern
    ## These Patterns will be assigned with positive values
    off_patterns = np.array([
        # Win Pattern:
        [1, 1, 1, 1, 1, 1],

        # Live 5: One more to go then the opponent will win
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1, 1],

        # Live 4: 1 more to go then the opponent will win
        [1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 1, 0],
        [1, 1, 1, 0, 0, 1],
        [1, 1, 0, 1, 1, 0],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 1],

        # Live 3: One more to become Live 4, 1 more to become Live 5
        [1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1]
    ])

    off_pattern_weight = np.array([
        WIN_WEIGHT,

        OFF_5_WEIGHT,
        OFF_5_WEIGHT,
        OFF_5_WEIGHT,

        OFF_4_WEIGHT,
        OFF_4_WEIGHT,
        OFF_4_WEIGHT,
        OFF_4_WEIGHT,
        OFF_4_WEIGHT,
        OFF_4_WEIGHT,
        OFF_4_WEIGHT,
        OFF_4_WEIGHT,

        OFF_3_WEIGHT,
        OFF_3_WEIGHT,
        OFF_3_WEIGHT,
        OFF_3_WEIGHT,
        OFF_3_WEIGHT,
        OFF_3_WEIGHT,
        OFF_3_WEIGHT
    ])

    ## Defensive Pattern
    ## These Patterns will be assigned with negative values
    def_patterns = off_patterns * 2
    def_pattern_weight = np.array([
        LOSE_WEIGHT,

        DEF_5_WEIGHT,
        DEF_5_WEIGHT,
        DEF_5_WEIGHT,

        DEF_4_WEIGHT,
        DEF_4_WEIGHT,
        DEF_4_WEIGHT,
        DEF_4_WEIGHT,
        DEF_4_WEIGHT,
        DEF_4_WEIGHT,
        DEF_4_WEIGHT,
        DEF_4_WEIGHT,

        DEF_3_WEIGHT,
        DEF_3_WEIGHT,
        DEF_3_WEIGHT,
        DEF_3_WEIGHT,
        DEF_3_WEIGHT,
        DEF_3_WEIGHT,
        DEF_3_WEIGHT
    ])
    return np.concatenate((off_patterns, def_patterns), axis=0), np.concatenate((off_pattern_weight, def_pattern_weight), axis=0)
    
def generate_all_orientations(patterns, weights):
    """Generate all mirrored and directional versions of each pattern."""
    all_patterns = []
    all_weights = []

    for pattern, weight in zip(patterns, weights):
        base = np.array(pattern)

        # Reshape to 1D horizontal pattern
        horiz = base.reshape(1, -1)

        # Vertical is transpose of horizontal
        vert = horiz.T
        
        # Diagonal (from top-left to bottom-right)
        diag_ = np.eye(len(base), dtype=int)
        diag_main = diag_ * horiz[0] - (1 - diag_)
        diag_main_flipped = np.fliplr(diag_main)

        # Add mirrored version
        horiz_flipped = np.fliplr(horiz)
        if (horiz_flipped == horiz).all():
            all_patterns.extend([
                horiz, vert, diag_main, diag_main_flipped
            ])

            all_weights.extend([weight for _ in range(4)])

        else:
            vert_flipped = horiz_flipped.T
            diag_anti = diag_ * horiz_flipped[0] - (1 - diag_)
            diag_anti_flipped = np.fliplr(diag_anti)

            # Add all to list
            all_patterns.extend([
                horiz, horiz_flipped,
                vert, vert_flipped,
                diag_main, diag_main_flipped,
                diag_anti, diag_anti_flipped
            ])

            all_weights.extend([weight for _ in range(8)])

    return all_patterns, all_weights

class Connect6ValueApproximator:

    def __init__(self, board_size=19):
        patterns, weights = pattern_generator()
        all_patterns, all_weights = generate_all_orientations(patterns, weights)

        # Convert patterns to 6x6 tensors (padded where needed)
        kernel_list = []
        for pattern in all_patterns:
            padded = torch.ones((6, 6), dtype=torch.float32) * -1
            h, w = pattern.shape
            padded[:h, :w] = torch.tensor(pattern.copy(), dtype=torch.float32)
            kernel_list.append(padded)

        self.kernels = torch.stack(kernel_list).unsqueeze(1)  # (N, 1, 6, 6)
        self.weights = torch.tensor(all_weights, dtype=torch.float32)

    def value(self, board):
        board_tensor = torch.tensor(board.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Extract all 6x6 sliding windows into flattened patches
        unfolded = F.unfold(board_tensor, kernel_size=(6, 6))  # (1, 36, L)
        unfolded = unfolded.squeeze(0).T  # (L, 36)

        kernels_flat = self.kernels.view(self.kernels.size(0), -1)  # (N, 36)

        # Compare all patterns against all patches (strict match)
        matches = (unfolded[None, :, :] == kernels_flat[:, None, :])  # (N, L, 36)
        # mask the don't-care positions (-1), make them true
        mask = (kernels_flat[:, None, :] == -1).expand(-1, matches.shape[1], -1)
        matches = matches | mask
        
        exact_matches = matches.all(dim=2).float()  # (N, L), 1 if full match

        match_counts = exact_matches.sum(dim=1)  # (N,)
        total_value = torch.dot(match_counts, self.weights)  # scalar

        return total_value.item()

    # def match_pattern(self, board, pattern, weight):
    #     """Check if a pattern exists anywhere in the board."""
    #     board = np.array(board)
    #     p_rows, p_cols = pattern.shape
    #     b_rows, b_cols = board.shape
        
    #     acc_value = 0

    #     for i in range(b_rows - p_rows + 1):
    #         for j in range(b_cols - p_cols + 1):
    #             sub_board = board[i:i+p_rows, j:j+p_cols]
    #             # A match occurs if all non-zero entries in the pattern match the board
    #             mask = (pattern != 0)
    #             if np.array_equal(sub_board[mask], pattern[mask]):
    #                 acc_value += weight

    #     return acc_value

    # def value(self, state):
    #     value = 0
    #     for pattern, weight in zip(self.all_patterns, self.all_weights):
    #         value += self.match_pattern(state, pattern, weight)
    #     return value

    def update(self):
        pass

def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: Connect6ValueNet instance.
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
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")

    return final_scores
