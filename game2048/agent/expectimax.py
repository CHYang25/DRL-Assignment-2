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

env = Game2048Env()

def create_env_from_state(state, score):
    sim_env = copy.deepcopy(env)
    sim_env.board = state.copy()
    sim_env.score = score
    return sim_env

MAX_DEPTH = 2

def get_tile_insertions(state):
    insertions = []
    empty_cells = [(i, j) for i in range(4) for j in range(4) if state[i][j] == 0]
    for (i, j) in empty_cells:
        for val, prob in [(2, 0.9), (4, 0.1)]:
            new_state = state.copy()
            new_state[i][j] = val
            insertions.append((new_state, prob))
    return insertions

def expectimax(approximator, state, score, legal_actions, depth=0):
    if create_env_from_state(state, score).is_game_over():
        return -100000, -1
    
    if depth == MAX_DEPTH or len(legal_actions) == 0:
        # print(score, approximator.value(state))
        return score + approximator.value(state), -1
    
    new_scores = []
    if depth % 2 == 0: # max node
        for action in legal_actions:
            sim_env = create_env_from_state(state, score)
            next_state, reward, done, _ = sim_env.step(action, False)
            legal_next_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            sc, at = expectimax(approximator, next_state, sim_env.score, legal_next_actions, depth=depth+1)
            new_scores.append(sc)
        return max(new_scores), legal_actions[np.argmax(new_scores)]
    else: # chance node
        insertions = get_tile_insertions(state)
        expected_score = 0
        for new_state, prob in insertions:
            sim_env = create_env_from_state(new_state, score)
            legal_next_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            sc, _ = expectimax(approximator, new_state, score, legal_next_actions, depth=depth+1)
            expected_score += prob * sc
        return expected_score / len(insertions), -1