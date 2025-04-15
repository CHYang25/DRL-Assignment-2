# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
from game2048.agent.ntuple_td_learning import NTupleApproximator
from game2048.game2048 import Game2048Env
from math import inf

env = Game2048Env()

def is_game_over(state):
    if np.any(state == 0):
        return False
    for i in range(state.shape[0]):
        for j in range(state.shape[1] - 1):
            if state[i, j] == state[i, j+1]:
                return False
    for j in range(state.shape[1]):
        for i in range(state.shape[0] - 1):
            if state[i, j] == state[i+1, j]:
                return False

    return True

def create_env_from_state(state, score):
    sim_env = copy.deepcopy(env)
    sim_env.board = state.copy()
    sim_env.score = score
    return sim_env

MAX_DEPTH = 4
INF = 2**64

def get_tile_insertions(state):
    insertions = []
    empty_cells = [(i, j) for i in range(4) for j in range(4) if state[i][j] == 0]
    for (i, j) in empty_cells:
        for val, prob in [(2, 0.9), (4, 0.1)]:
            new_state = state.copy()
            new_state[i][j] = val
            insertions.append((new_state, prob))
    return insertions

def expectimax(approximator, state, score, legal_actions, depth=0, max_depth = MAX_DEPTH):
    if is_game_over(state):
        return -INF, -1
    
    if depth == max_depth or len(legal_actions) == 0:
        # print(score, approximator.value(state))
        return approximator.value(state), -1
    
    new_scores = []
    if depth % 2 == 0: # max node
        for action in legal_actions:
            sim_env = create_env_from_state(state, score)
            next_state, reward, done, _ = sim_env.step(action, False)
            legal_next_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            sc, at = expectimax(approximator, next_state, sim_env.score, legal_next_actions, depth=depth+1, max_depth=max_depth)
            new_scores.append(sc)
        return max(new_scores), legal_actions[np.argmax(new_scores)]
    else: # chance node
        insertions = get_tile_insertions(state)
        expected_score = 0
        for new_state, prob in insertions:
            sim_env = create_env_from_state(new_state, score)
            legal_next_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            sc, _ = expectimax(approximator, new_state, score, legal_next_actions, depth=depth+1, max_depth=max_depth)
            expected_score += prob * sc
        if len(insertions) > 0:
            expected_score /= len(insertions)
        return expected_score, -1
    

if __name__ == '__main__':
    from game2048.agent.heuristic_mcts import HEU_Approximator

    env = Game2048Env()

    approximator = HEU_Approximator()

    final_score = []
    for episode in range(10):
        state = env.reset()
        env.render()

        done = False
        while not done:
            # Select the best action (based on highest visit count)
            legal_actions = [a for a in range(4) if env.is_move_legal(a)]
            score, action = expectimax(approximator, state, env.score, legal_actions, depth=0, max_depth=3)
            print("Expectimax selected action:", action, "Score:", env.score)

            # Execute the selected action and update the state
            state, reward, done, _ = env.step(action)
            print("State:\n", state)
            # env.render(action=action)

        print("Game over, final score:", env.score)
        final_score.append(env.score)
        
    print("Average score over 10 episodes:", np.mean(final_score))