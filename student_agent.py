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

MAX_DEPTH = 1

def expectimax(state, score, legal_actions, depth=0):
    if depth == MAX_DEPTH or len(legal_actions) == 0:
        # print(score, approximator.value(state))
        return score + approximator.value(state), -1
    
    new_scores = []
    if depth % 2 == 0: 
        for action in legal_actions:
            sim_env = create_env_from_state(state, score)
            next_state, reward, done, _ = sim_env.step(action)
            legal_next_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            sc, at = expectimax(next_state, sim_env.score, legal_next_actions, depth=depth+1)
            new_scores.append(sc)
        return max(new_scores), legal_actions[np.argmax(new_scores)]
    else:
        for action in legal_actions:
            sim_env = create_env_from_state(state, score)
            next_state, reward, done, _ = sim_env.step(action)
            legal_next_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            sc, at = expectimax(next_state, sim_env.score, legal_next_actions, depth=depth+1)
            new_scores.append(sc)
        return sum(new_scores) / len(new_scores), -1


def get_action(state, score):
    sim_env = create_env_from_state(state, score)
    legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
    new_score, action = expectimax(state, score, legal_actions)
    return action


if __name__ == '__main__':
    state = env.reset()
    done = False
    cnt = 0

    while not done:

        action = get_action(state, env.score)
        
        print("Step:", cnt, "Score:", env.score)
        print("State:\n", state)

        state, reward, done, _ = env.step(action)
        cnt += 1