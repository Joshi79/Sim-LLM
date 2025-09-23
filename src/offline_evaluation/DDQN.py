###### IMPORTS ######
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


###### HYPERPARAMETERS ######
GAMMA = 0.9
TAU = 1e-4 # 1e-4

seed = 1

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda")



######################### DEFINING NETWORKS #########################    
class DDQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size):
        super(DDQN, self).__init__()
        self.head_1 = nn.Linear(state_size, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)
        self.ln1 = nn.LayerNorm(layer_size)
        self.ln2 = nn.LayerNorm(layer_size)

    def forward(self, input):
        x = torch.relu(self.ln1(self.head_1(input)))
        x = torch.relu(self.ln2(self.ff_1(x)))
        out = self.ff_2(x)
        
        return out


######################### DEFINING RL AGENT #########################
class DDQNAgent():
    def __init__(self, state_size, action_size, learning_rate, hidden_size, device= torch.device("cpu"),):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.tau = TAU
        self.gamma = GAMMA
        
        self.network = DDQN(state_size=self.state_size,
                            action_size=self.action_size,
                            layer_size=hidden_size
                            ).to(self.device)
        
        
        self.target_net = DDQN(state_size=self.state_size,
                            action_size=self.action_size,
                            layer_size=hidden_size
                            ).to(self.device)
        
        self.optimizer = optim.Adam(params=self.network.parameters(), lr=learning_rate) 
            
    @torch.jit.export
    def select_action_batch(self, state):
        with torch.no_grad():
            action_probs = self.network(state) # torch one hot encoding
            _, max_idx = torch.max(action_probs, dim=-1)

            # Use torch.nn.functional.one_hot to create one-hot encoded tensor
            one_hot_action = F.one_hot(max_idx, num_classes=action_probs.size(-1))

            return one_hot_action
        
    def select_action_batch_not_encoded(self, state):
        with torch.no_grad():
            action_probs = self.network(state) # torch one hot encoding
            _, max_idx = torch.max(action_probs, dim=-1)

            return max_idx
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        


    @torch.jit.export
    def train(self, replay_buffer):
        
        # Sample replay buffer
        state, action, next_state, reward, _, _ = replay_buffer.sample()

        with torch.no_grad():
            next_actions = self.network(next_state).argmax(1, keepdim=True)
            Q_targets_next = self.target_net(next_state).gather(1, next_actions)
            Q_targets = reward + (self.gamma * Q_targets_next)
            

        Q_a_s = self.network(state)
        Q_expected = torch.gather(Q_a_s, 1, action)


        bellman_error = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()#set_to_none=True
        

        bellman_error.backward()
        clip_grad_norm_(self.network.parameters(), 1.)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_net)

        # Don't return loss with item to improve speed
        return bellman_error.detach().item()