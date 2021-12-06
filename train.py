import time
import random
import math
from collections import deque

import numpy as np
# from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch

from mlagents_envs.environment import UnityEnvironment
from dqn_agent import DQNAgent
from model import DQNLinear
from mla_wrapper_sa import MLA_Wrapper
from cnn import CNN

env = MLA_Wrapper()

env.reset()
# behavior_name = list(env.behavior_specs)[0] 
# spec = env.behavior_specs[behavior_name]

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device: ", device)
# print("Number of observations : ", len(spec.observation_specs))
# print(f"There are {spec.action_spec.discrete_size} discrete actions")

STATE_SIZE = env.obs_shape
ACTION_SIZE = env.action_space[0].n
GAMMA = 0.99           # discount factor
BUFFER_SIZE = int(1e5)      # replay buffer size
BATCH_SIZE = 32        # Update batch size
LR = 1e-3              # learning rate 
TAU = 1e-2             # for soft update of target parameters
UPDATE_EVERY = 5       # how often to update the network 

agent = DQNAgent(STATE_SIZE, ACTION_SIZE, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, device)
# agent = CNN(STATE_SIZE, output_size=ACTION_SIZE)
EPS_START = 0.99       # starting value of epsilon
EPS_END = 0.01         # Ending value of epsilon
EPS_DECAY = 200       # Rate by which epsilon to be decayed

epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx / EPS_DECAY)

# plt.plot([epsilon_by_epsiode(i) for i in range(2000)])
# plt.show()

def plot_result(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def train(n_episodes, scores_average_window, benchmark_reward):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
        benchmark_reward (float): benchmark reward at which environment is solved.
    """
    scores = []
    scores_window = deque(maxlen=SCORES_AVERAGE_WINDOW)
    state = env.reset()

    for i_episode in range(1, NUM_EPISODES+1):
        score = 0
        eps = epsilon_by_epsiode(i_episode)
        while True:
            #ds, ts = env.get_steps(behavior_name=behavior_name)
            action = agent.act(state, eps)
            next_state, reward, done, info, masks = env.step(action)
            score += reward
            agent.step(state, action, reward, next_state, done, masks)
            state = next_state
            if np.any(done):
                break
            
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        
        # printing and ploting results
        # clear_output(wait=True)
        # plot_result(scores)
        print('\rEpisode {}\tAverage Score: {:.2f}\tEplision: {:.3f}'.format(i_episode, np.mean(scores_window), eps))
        
        if float(np.mean(scores_window)) >= BENCHMARK_REWARD:
            agent.save_model("basic_solved.pth", scores)
            print("Yah Environment is solved :)")
            break
    
    return scores
BENCHMARK_REWARD = 0.9300
SCORES_AVERAGE_WINDOW = 100
NUM_EPISODES = 2000

scores = train(NUM_EPISODES, SCORES_AVERAGE_WINDOW, BENCHMARK_REWARD)
print("Done Training")

env.close()