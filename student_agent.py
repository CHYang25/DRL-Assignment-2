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
from game2048.agent.heuristic_mcts import HEU_MCTS, HEU_MCTS_Node

random.seed(42)
np.random.seed(42)

heu_mcts = HEU_MCTS(Game2048Env(), iterations=5, exploration_constant=0, rollout_depth=3, gamma=0.99)
# td_mcts = TD_MCTS(
#     Game2048Env(), 
#     pickle.load(open("./game2048/agent/DRL-Assignment-2-Checkpoint/n-tuple-approximator.pkl", "rb")), 
#     iterations=5, exploration_constant=0, rollout_depth=3, gamma=0.99)

def get_action(state, score):
    env = heu_mcts.create_env_from_state(state, score)
    root = HEU_MCTS_Node(env, state, score)

    # heu_mcts.iterations = len(root.untried_actions) + 1
    for _ in range(heu_mcts.iterations):
        heu_mcts.run_simulation(root)

    # Select the best action (based on highest visit count)
    best_act, _ = heu_mcts.best_action_distribution(root)

    return best_act

if __name__ == '__main__':
    env = Game2048Env()

    state = env.reset()
    done = False
    cnt = 0

    while not done:

        action = get_action(state, env.score)
        
        print("Step:", cnt, "Score:", env.score, "Action:", action)

        state, reward, done, _ = env.step(action)
        cnt += 1

        print("State:\n", state)