from math import exp
from mlagents_envs.base_env import ActionSpec
import numpy as np
from numpy.random import rand
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from buffer import ReplayBuffer
from model import DQNLinear
from cnn import CNN
class DQNAgent():
    def __init__(self, input_shape, action_size, buffer_size, batch_size, gamma, lr, tau, update_every, device):
        """Initialize an Agent object.
        
        Params
        ======
            input_shape (tuple): dimension of each state
            action_size (int): dimension of each action
            buffer_size (int): replay buffer size
            batch_size (int):  minibatch size
            gamma (float): discount factor
            lr (float): learning rate 
            tau (float): Soft-parameter update
            update_every (int): how often to update the network
            device(string): Use Gpu or CPU
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_every = update_every
        self.tau = tau
        self.device = device
     # Q-Network
        self.policy_net = DQNLinear(input_shape, action_size).to(self.device)
        self.target_net = DQNLinear(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.device)
        
        self.t_step = 0

    
    def step(self, state, action, reward, next_state, done, masks):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, masks)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory.memory[0][0]) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
                
    def act(self, state, eps=0.01):
        states=torch.from_numpy(state).to(self.device)
        actions = torch.zeros(states.shape[:2])
        self.policy_net.eval()
        with torch.no_grad():
            for i in range(len(state)):
                state = states[i]
                action_values = self.policy_net(state)
                action = torch.max(action_values, dim=1)[1]
                actions[i] = action

        self.policy_net.train()
        
        for rolls in range(len(actions)):
            for agent in range(len(actions[rolls])):
                if random.random() < eps:
                    actions[rolls, agent] = random.choice(np.arange(self.action_size))

        # # Epsilon-greedy action selection
        # if random.random() > eps:
        #     return action_values.cpu().data.numpy()
        # else:
        #     return random.choice(np.arange(self.action_size))
        return actions.cpu().data.numpy()
        
    def learn(self, experiences):
        states_all, actions_all, rewards_all, next_states_all, dones_all = experiences
        states, actions, rewards, next_states, dones = experiences
        # shapes = experiences[0].shape
        # n_rolls = shapes[1]
        # n_agents = shapes[2]
        # for r in range(n_rolls):
        #     for a in range(n_agents):
        #         random_r = random.choice(np.arange(n_rolls))
        #         random_a = random.choice(np.arange(n_agents))
        #         random_r = r
        #         random_a = a
        #         states = states_all[:,random_r, random_a]
        #         actions = actions_all[:, random_r, random_a]
        #         rewards = rewards_all[:, random_r, random_a].squeeze()
        #         next_states = next_states_all[:, random_r, random_a]
        #         dones = dones_all[:, random_r, random_a]

            # Get expected Q values from policy model
        Q_expected_current = self.policy_net(states)
        Q_expected = Q_expected_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get max predicted Q values (for next states) from target model
        # Q_targets_next = self.target_net(next_states).detach().max(1)[0]
        
        # # Compute Q targets for current states 
        # Q_targets = rewards.squeeze() + (self.gamma * Q_targets_next * (1 - dones))
        
        argmaxA = torch.max(self.policy_net(next_states), dim=1)[1]
        targetQ = self.target_net(next_states).detach().gather(1, argmaxA.unsqueeze(1)).squeeze(1)
        Q_targets = rewards.squeeze() + (self.gamma * targetQ * (1 - dones))
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.policy_net, self.target_net, self.tau)

    
    # θ'=θ×τ+θ'×(1−τ)
    def soft_update(self, policy_model, target_model, tau):
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)

    def load_model(self, path):
        
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['state_dict'])
        self.target_net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        scores = checkpoint['scores']

        return scores

    def save_model(self, path, scores):
        model = {
            "state_dict": self.policy_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scores": scores
        }
        torch.save(model, path)