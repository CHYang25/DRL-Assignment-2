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

approximator = pickle.load(open("./game2048/agent/DRL-Assignment-2-Checkpoint/n-tuple-approximator.pkl", "rb"))

env = Game2048Env()

def create_env_from_state(state, score):
    sim_env = copy.deepcopy(env)
    sim_env.board = state.copy()
    sim_env.score = score
    return sim_env

MAX_DEPTH = 10

def expectimax(state, score, depth=0):
    if depth == MAX_DEPTH:
        return score + approximator.value(state)
    
    new_scores = []
    if depth % 2 == 0: 
        for action in range(4):
            sim_env = create_env_from_state(state, score)
            next_state, reward, done, _ = sim_env.step(action)
            new_scores.appned(expectimax(next_state, reward + score, depth=depth+1))
        return max(new_scores), np.argmax(new_scores)
    else:
        for action in range(4):
            sim_env = create_env_from_state(state, score)
            next_state, reward, done, _ = sim_env.step(action)
            new_scores.append(expectimax(next_state, reward + score, depth=depth+1))
        return sum(new_scores) / 4, -1


def get_action(state, score):
    new_score, action = expectimax(state, score)
    return action


if __name__ == '__main__':
    print(approximator)