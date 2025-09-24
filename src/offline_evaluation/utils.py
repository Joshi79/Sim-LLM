import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
import os
from copy import deepcopy




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLUMNS_RL_ALGORITHM = ['day_part_x', 'numberRating', 'highestRating', 'lowestRating', 'medianRating', 'sdRating', 'numberLowRating', 'numberMediumRating', 'numberHighRating',
                    'numberMessageReceived', 'numberMessageRead', 'readAllMessage']
ACTION_LABELS = ['No message', 'Encouraging', 'Informing', 'Affirming']


# Generic replay buffer for standard gym tasks
class ReplayBuffer(object):
	def __init__(self, state_dim, batch_size, buffer_size, device):
		self.batch_size = batch_size
		self.max_size = int(buffer_size)
		self.device = device

		self.ptr = 0 # pointer next insertion
		self.crt_size = 0

		self.state = torch.zeros((self.max_size, state_dim), dtype=torch.float32, device=device)
		self.action = torch.zeros((self.max_size, 1), dtype=torch.int64, device=device) # so not one-hot encode here
		self.next_state = torch.zeros((self.max_size, state_dim), dtype=torch.float32, device=device)
		self.reward = torch.zeros((self.max_size, 1), dtype=torch.float32, device=device)
		self.done = torch.zeros((self.max_size, 1), dtype=torch.float32, device=device)
		self.user_id = torch.zeros((self.max_size, 1), dtype=torch.float32, device=device)


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = torch.tensor(state, dtype=torch.float32, device=self.device)
		self.action[self.ptr] = torch.tensor(action, dtype=torch.int64, device=self.device)
		self.next_state[self.ptr] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
		self.reward[self.ptr] = torch.tensor(reward, dtype=torch.float32, device=self.device)
		self.done[self.ptr] = torch.tensor(done, dtype=torch.float32, device=self.device)

		self.ptr = (self.ptr + 1) % self.max_size
		self.crt_size = min(self.crt_size + 1, self.max_size)


	def sample(self):
		ind = np.random.randint(0, self.crt_size, size=self.batch_size)
		state = self.state[ind]
		action = self.action[ind]
		next_state = self.next_state[ind]
		reward = self.reward[ind]
		done = self.done[ind]
		user_id = self.user_id[ind]
		return state, action, next_state, reward, done, user_id



	def save(self, save_folder):
		np.save(f"{save_folder}_state.npy", self.state[:self.crt_size])
		np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
		np.save(f"{save_folder}_next_state.npy", self.next_state[:self.crt_size])
		np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
		np.save(f"{save_folder}_done.npy", self.done[:self.crt_size])
		np.save(f"{save_folder}_ptr.npy", self.ptr)


	def load(self, save_folder, size=-1):
		reward_buffer = np.load(f"{save_folder}_reward.npy")
		
		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.crt_size = min(reward_buffer.shape[0], size)

		self.state[:self.crt_size] = np.load(f"{save_folder}_state.npy")[:self.crt_size]
		self.action[:self.crt_size] = np.load(f"{save_folder}_action.npy")[:self.crt_size]
		self.next_state[:self.crt_size] = np.load(f"{save_folder}_next_state.npy")[:self.crt_size]
		self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
		self.done[:self.crt_size] = np.load(f"{save_folder}_done.npy")[:self.crt_size]

		print(f"Replay Buffer loaded with {self.crt_size} elements.")


	def load_d4rl_dataset(self, data):
		if self.crt_size != 0:
			raise ValueError("Trying to load data into non-empty replay buffer")
		size = data["states"].shape[0]
		if size > self.max_size:
			raise ValueError(
				"Replay buffer is smaller than the dataset you are trying to load!"
			)
		
		self.crt_size = size

		# Fill states and next states for rl algorithm
		self.state[:size] = torch.tensor(data["states"], dtype=torch.float32, device=device)
		self.next_state[:size] = torch.tensor(data["next_states"], dtype=torch.float32, device=device)

		# Fill actions, rewards, and dones
		self.action[:size] =torch.tensor(data["actions"].reshape(-1, 1), dtype=torch.float32, device=device)
		self.reward[:size] = torch.tensor(data["rewards"].reshape(-1, 1), dtype=torch.float32, device=device)
		self.done[:size] = torch.tensor(data["terminals"].reshape(-1, 1), dtype=torch.float32, device=device)

		self.user_id[:size] = torch.tensor(data["user_ids"].reshape(-1, 1), dtype=torch.float32, device=device)

		# If n_transitions is smaller than _size, it means the loaded data fills only a portion of the buffer, so _pointer should be set to n_transitions.
		self.ptr = size



def get_format_data_rl_algorithm(df):


    # Nr of columns included in state for rl algorithm and fqe
    nr_columns_rl_algorithm = len(COLUMNS_RL_ALGORITHM)

    # Sort based on user_id and timestamp for sanity
    df = df.sort_values(['user_id','serverTimestamp', 'day_part_x'])

    # List for next states fqe and rl algorithm, and terminal states
    next_states = []
    terminal_states_list = []

    # List user_ids's
    list_userids = []

    # For getting next state
    # Loop over depression_df per user_id to also get terminal flag
    grouped = df.groupby('user_id')
    for user_id, user_id_df in grouped:

        # Reset index
        user_id_df = user_id_df.reset_index(inplace=False)

        # Loop over df specific user
        for i, (index, row) in enumerate(user_id_df.iterrows()):
            # Append user_id
            list_userids.append(user_id)

            # Set terminal state
            terminal_state = False
            # Check if last row, then terminal flag is 1, otherwise 0
            if index == len(user_id_df) - 1:
                terminal_state = True

            # If terminal state, next state only has zeros
            if terminal_state:
                row_values_next_state = nr_columns_rl_algorithm * [0]
            # If state isn't terminal, get values next row
            else:
                # Row values next state
                row_values_next_state_df = user_id_df.loc[[index+1]]
                # Values for specific columns rl algorithm and fqe
                row_values_next_state = list(row_values_next_state_df[COLUMNS_RL_ALGORITHM].iloc[0])

            # Append next state to lists
            next_states.append(row_values_next_state)

            # We don't have terminal states
            terminal_states_list.append(0)


    # Convert user_id to number
    # Dictionary to map strings to unique numbers
    string_to_number = {}

    # Counter for unique numbers
    counter = 0

    # Loop through the list and replace strings with numbers
    for i in range(len(list_userids)):
        if list_userids[i] not in string_to_number:
            string_to_number[list_userids[i]] = counter
            counter += 1
        list_userids[i] = string_to_number[list_userids[i]]

    

    # Get states, actions, reward, next states, and terminals in numpy array
    # Get states values for fqe and rl algorithm
    states = df[COLUMNS_RL_ALGORITHM].to_numpy()
    # Reward
    rewards = df['reward'].to_numpy()
    # Next states fqe and rl algorithm
    next_states = np.array(next_states)
    terminal_states_list = np.array(terminal_states_list)

    # User id list to np array
    list_userids = np.array(list_userids)

    # One-hot encoding
    # Get actions in right format
    actions = df['action'].to_numpy()
    actions_array = []
    for action in actions:
        actions_array.append(action)
    actions = actions_array

    
    # Numpy array van actions en prev actions??
    actions = np.array(actions)

    # Create dictionary based on 'observations', 'actions', 'next_observations', 'terminals'
    dict_dataset = {'states': states, 'actions': actions,
                    'next_states': next_states, 'rewards': rewards, 'terminals': terminal_states_list, 
                    'user_ids': list_userids}
    
    return dict_dataset
class FixedAgent:
    """Fixed policy that always selects the same action"""

    def __init__(self, fixed_action=0, device="cpu"):
        self.fixed_action = int(fixed_action)
        self.device = device
        self.action_size = 4
        self.network = None  # compatibility

    def select_action_batch_not_encoded(self, states):
        batch_size = int(states.shape[0])
        return torch.full((batch_size,), self.fixed_action, dtype=torch.long, device=self.device)

    def select_action_batch(self, states):
        idx = self.select_action_batch_not_encoded(states)
        return F.one_hot(idx, num_classes=self.action_size).to(dtype=torch.float32)

    def select_action(self, state):
        return self.fixed_action

def _split_episodes(df, id_col="user_id"):
    """Return a list of dataframes, one per episode (grouped by user_id)."""
    return [g.reset_index(drop=True) for _, g in df.groupby(id_col, observed=False)]


def _build_rb(df_ep,rb_cls,state_dim=12,batch_size=64,device= "cpu"):
    """Build replay buffer from dataframe"""

    d = get_format_data_rl_algorithm(df_ep)
    rb = rb_cls(state_dim=state_dim,
                batch_size=batch_size,
                buffer_size=len(df_ep) + 1,
                device=device)
    rb.load_d4rl_dataset(d)
    return rb


def extended_stats(samples):
    """Compute extended statistics for bootstrap samples"""
    s = samples
    return {
        "n_samples": int(len(s)),
        "mean": float(np.mean(s)),
        "std": float(np.std(s)),
        "median": float(np.median(s)),
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "range": float(np.max(s) - np.min(s)),
        "q25": float(np.percentile(s, 25)),
        "q75": float(np.percentile(s, 75)),
        "iqr": float(np.percentile(s, 75) - np.percentile(s, 25)),
        "skewness": float(pd.Series(s).skew()),
        "kurtosis": float(pd.Series(s).kurtosis()),
        "se": float(np.std(s) / np.sqrt(len(s))),
        "cv": float(np.std(s) / abs(np.mean(s))) if np.mean(s) != 0 else float('inf')
    }
def summarize_checkpoint_for_json(checkpoint,config,policy_type):
    if policy_type == "learned":
        h = checkpoint.get("hyperparameters", {})
        return {
            "model_info": {
                "architecture": "DDQN",
                "state_size": 12,
                "action_size": 4,
                "hidden_size": int(h.get("hidden_size", 128))
            },
            "training_info": {
                "final_fqe": checkpoint.get("final_fqe", "N/A"),
                "dataset": checkpoint.get("dataset", "N/A"),
                "training_epochs": checkpoint.get("training_epochs", "N/A")
            },
            "hyperparameters": {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in h.items()},
            "agent_path": os.path.abspath(config.get("agent_path", "")),
        }
    else:
        h = checkpoint.get("hyperparameters", {})
        return {
            "model_info": {
                "architecture": "fixed_policy",
                "state_size": 12,
                "action_size": 4,
                "hidden_size": None
            },
            "training_info": {
                "final_fqe": "N/A",
                "dataset": "N/A",
                "training_epochs": "N/A"
            },
            "hyperparameters": h,
            "agent_path": None
        }

def set_global_seeds(seed):
    """Set seed for repocubility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def compute_policy_action_distribution(agent, df):
    """Compute action distribution for a policy."""
    formatted = get_format_data_rl_algorithm(df)
    states = torch.tensor(
        formatted["states"],
        dtype=torch.float32,
        device=getattr(agent, "device", "cpu"),
    )
    with torch.no_grad():
        pred_idx = agent.select_action_batch_not_encoded(states)

    pred_idx_cpu = pred_idx.detach().cpu().numpy()
    return {ACTION_LABELS[i]: int((pred_idx_cpu == i).sum()) for i in range(len(ACTION_LABELS))}


def compute_point_estimate(df, agent, device="cpu"):
    # Lokaler Import vermeidet Zirkularimport (FQE_def -> utils)
    from .FQE_def import perform_FQE

    d = get_format_data_rl_algorithm(df)
    rb_full = ReplayBuffer(state_dim=12, batch_size=64, buffer_size=len(df) + 1, device=device)
    rb_full.load_d4rl_dataset(d)
    agent_copy = deepcopy(agent)
    v_list, _ = perform_FQE(rb_full, agent_copy, df, bool_learned_policy=True)
    return float(v_list[-1])

def load_trained_agent(agent_path, device= "cpu"):
    """Load trained DDQN which was prebiously tuned with Optuna"""

    from .DDQN import DDQNAgent

    map_location = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    checkpoint = torch.load(agent_path, map_location=map_location, weights_only=False)

    hyperparams = checkpoint['hyperparameters']

    agent = DDQNAgent(
        state_size=12,
        action_size=4,
        learning_rate=hyperparams.get('learning_rate', 0.001),
        hidden_size=hyperparams.get('hidden_size', 128),
        device=device
    )

    agent.tau = hyperparams.get('tau', 0.005)
    agent.gamma = hyperparams.get('gamma', 0.99)

    agent.network.load_state_dict(checkpoint['model_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])

    if device == "cuda" and torch.cuda.is_available():
        agent.network = agent.network.to(device)
        agent.target_net = agent.target_net.to(device)

    return agent, checkpoint


def load_policy(config, device="cpu"):
	"""Load either a learned DDQN agent or create a fixed policy agent"""

	if config["policy_type"] == "learned":
		return load_trained_agent(config["agent_path"], device)

	elif config["policy_type"] == "fixed":
		agent = FixedAgent(fixed_action=config["fixed_action"], device=device)
		checkpoint = {
			"hyperparameters": {
				"policy_type": "fixed",
				"fixed_action": config["fixed_action"]
			}
		}
		return agent, checkpoint
	else:
		raise ValueError(f"Unknown policy type: {config['policy_type']}")


