
import numpy as np
import random
from collections import namedtuple, deque
import torch

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device, rolls=32, agents=1):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            device (string): GPU or CPU
        """

        self.memory = [[deque(maxlen=int(buffer_size/rolls/agents)) for agent in range(agents)] for roll in range(rolls)]
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device
        self.rolls = rolls
        self.agents = agents
    
    def add(self, state, action, reward, next_state, done, masks):
        for i in masks:
            roll = i[0]
            agent = i[1]
            e = self.experience(state[roll, agent], action[roll, agent], reward[roll, agent], next_state[roll, agent], done[roll, agent])
            self.memory[roll][agent].append(e)
    
    def sample(self):
        experiences = []
        for i in range(self.rolls):
            for j in range(self.agents):
                cur_experiences = random.sample(self.memory[i][j],k=self.batch_size)
                experiences += cur_experiences
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)