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
from game2048.agent.td_mcts import TD_MCTS, TD_MCTS_Node

env = Game2048Env()

approximator = pickle.load(open("./game2048/agent/DRL-Assignment-2-Checkpoint/n-tuple-approximator.pkl", "rb"))
td_mcts = TD_MCTS(env, approximator, iterations=100, exploration_constant=1.41, rollout_depth=10, gamma=0.99)


# def create_env_from_state(state, score):
#     sim_env = copy.deepcopy(env)
#     sim_env.board = state.copy()
#     sim_env.score = score
#     return sim_env

# MAX_DEPTH = 4

# def get_tile_insertions(state):
#     insertions = []
#     empty_cells = [(i, j) for i in range(4) for j in range(4) if state[i][j] == 0]
#     for (i, j) in empty_cells:
#         for val, prob in [(2, 0.9), (4, 0.1)]:
#             new_state = state.copy()
#             new_state[i][j] = val
#             insertions.append((new_state, prob))
#     return insertions

# def expectimax(state, score, legal_actions, depth=0):
#     if create_env_from_state(state, score).is_game_over():
#         return -100000, -1
    
#     if depth == MAX_DEPTH or len(legal_actions) == 0:
#         # print(score, approximator.value(state))
#         return score + approximator.value(state), -1
    
#     new_scores = []
#     if depth % 2 == 0: # max node
#         for action in legal_actions:
#             sim_env = create_env_from_state(state, score)
#             next_state, reward, done, _ = sim_env.step(action, False)
#             legal_next_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
#             sc, at = expectimax(next_state, sim_env.score, legal_next_actions, depth=depth+1)
#             new_scores.append(sc)
#         return max(new_scores), legal_actions[np.argmax(new_scores)]
#     else: # chance node
#         insertions = get_tile_insertions(state)
#         expected_score = 0
#         for new_state, prob in insertions:
#             sim_env = create_env_from_state(new_state, score)
#             legal_next_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
#             sc, _ = expectimax(new_state, score, legal_next_actions, depth=depth+1)
#             expected_score += prob * sc
#         return expected_score / len(insertions), -1


# def get_action(state, score):
#     sim_env = create_env_from_state(state, score)
#     legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
#     new_score, action = expectimax(state, score, legal_actions)
#     return action

def get_action(state, score):
    root = TD_MCTS_Node(state, score)

    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    # Select the best action (based on highest visit count)
    best_act, _ = td_mcts.best_action_distribution(root)

    return best_act

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