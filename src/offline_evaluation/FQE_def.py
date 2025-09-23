import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.offline_evaluation.utils import COLUMNS_RL_ALGORITHM
import copy
import time
from torch.nn.utils import clip_grad_norm_

# https://github.com/BY571/CQL/tree/main
# https://github.com/theophilegervet/discrete-off-policy-evaluation/blob/master/fqe.py

seed = 1

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


### FIXED HYPERPARAMETERS ###
DISCOUNT_FACTOR_FQE = 0.9
HIDDEN_SIZE_FCN_FQE = 64 # 64
NUMBER_LAYERS_FCN_FQE = 3
LEARNING_RATE_FQE = 4e-3 # 1e-4#0.0001
#BATCH_SIZE_FQE = 16 #1 # 64 # 128
N_STEPS_FQE = 60000


STATE_SIZE = 12

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TAU = 9e-3

BATCH_SIZE_RANDOM_POLICY = 32
NUM_CLASSES_RANDOM_POLICY = 4

# Order based on timestamp and user_id, reset index
def order_df(df):
    # Reset the index
    df = df.reset_index(drop=True, inplace=False)
    
    # Transform timestamp into correct type and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df.sort_values(['user_id','timestamp'])

    # Reset the index
    df = df.reset_index(drop=True, inplace=False)

    return df

# Get avg value starting states
def get_avg_value_start_states(FQE_Q_network, network_learned_policy, df, bool_learned_policy):
    """
    Get average value start states
    :param FQE_Q_network: FQE Q-network to get Q values from
    :param network_learned_policy: if not a fixed policy is used, give network learned policy
    :param df: df to get values from
    """
    # Order df for sanitiy
    df = order_df(df)
    # Group by user_id and extract the first row for each group
    first_row_per_user = df.groupby('user_id', observed=False).first().reset_index()
    # State rl algorithm
    states = first_row_per_user[COLUMNS_RL_ALGORITHM].to_numpy()   
 
    states_rl = torch.tensor(states,dtype=torch.float32, device=device)



    # Get policy if using RL and not fixed policies
    if bool_learned_policy:
        with torch.no_grad():
            policy = network_learned_policy.select_action_batch(states_rl)
    else:
        policy = network_learned_policy



    # Determine value start states with policy
    with torch.no_grad():
        # Get value each start state
        values_start_states = torch.sum(FQE_Q_network(states_rl) * policy, 1)
        # Get mean value start states
        mean_value_start_states = torch.mean(values_start_states).item()
    

    # Get mean value over all states
    return mean_value_start_states



# Q-network for getting Q(s,a) for FQE
class FQE_QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, nr_layers: int, layer_dimension: int):
        super(FQE_QNetwork, self).__init__()
  
        if nr_layers == 3:
            self.net = nn.Sequential(
            nn.Linear(state_dim, layer_dimension),
            nn.LayerNorm(layer_dimension),
            nn.ReLU(),
            nn.Linear(layer_dimension, layer_dimension),
            nn.LayerNorm(layer_dimension),
            nn.ReLU(),
            nn.Linear(layer_dimension, action_dim)
        )
            

    #@torch.compile
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
    

  
# Def for getting FQE specific policy
def perform_FQE(replay_buffer, network_learned_policy, df, bool_learned_policy):
    """
    Performs fitted Q-evaluation

    :param replay_buffer: the replaybuffer to sample from
    :param network_learned_policy: the network of the learned policy if applicable
    :param df: df for performing FQE on
    :return: returns FQE value policy
    """

    # Create FQE_Q network and optimizer
    FQE_Q_network = FQE_QNetwork(state_dim=STATE_SIZE, action_dim=4, nr_layers=NUMBER_LAYERS_FCN_FQE, layer_dimension=HIDDEN_SIZE_FCN_FQE) # Nr of actions is 4, one hot encoding thus dimension is 4
    FQE_Q_network = FQE_Q_network.to(device)

    # Initialize target network
    target_FQE_Q_network = copy.deepcopy(FQE_Q_network)

    # Optimizer
    optimizer = torch.optim.Adam(FQE_Q_network.parameters(), LEARNING_RATE_FQE) # was , weight_decay=1e-2

    # List for saving for policy in for loop FQE, nr evaluations per calculating FQE/loss, loss
    FQE_values_policy = []

    # Loss list
    list_loss = []

    # Loop over n_steps with stepsize of batch_size
    for step in range(0, N_STEPS_FQE): 
        """# If we do on the test set we want to see the convergence, so we save it
        if bool_test:
            if step % 10000 ==0:
                FQE_values_policy.append(get_avg_value_start_states(FQE_Q_network, network_learned_policy, df, bool_learned_policy))
                print(FQE_values_policy)"""
    
        states, actions, next_states, rewards, _, _ = replay_buffer.sample()


        if bool_learned_policy:
            # Get action learned policies if applicable
            policy = network_learned_policy.select_action_batch(next_states)
        else:
            policy = network_learned_policy
    
        with torch.no_grad():
            #print('target_FQE_Q_network(next_states) ',target_FQE_Q_network(next_states))
            target_Q = target_FQE_Q_network(next_states) * policy
            target_Q = rewards + (DISCOUNT_FACTOR_FQE * target_Q.sum(dim=1).view(-1, 1))
            

            
        Q_a_s = FQE_Q_network(states)
        Q_expected = torch.gather(Q_a_s, 1, actions)

        bellman_error = F.mse_loss(Q_expected, target_Q)

        optimizer.zero_grad()#set_to_none=True

        bellman_error.backward()
        clip_grad_norm_(FQE_Q_network.parameters(), 1.)
        optimizer.step()

        # ------------------- update target network ------------------- #
        for target_param, param in zip(target_FQE_Q_network.parameters(), FQE_Q_network.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        list_loss.append(bellman_error.detach().item())

    # Get FQE 
    FQE_values_policy.append(get_avg_value_start_states(FQE_Q_network, network_learned_policy, df, bool_learned_policy))
    return FQE_values_policy, list_loss
