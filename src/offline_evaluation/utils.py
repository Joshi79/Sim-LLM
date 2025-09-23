import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLUMNS_RL_ALGORITHM = ['day_part_x', 'numberRating', 'highestRating', 'lowestRating', 'medianRating', 'sdRating', 'numberLowRating', 'numberMediumRating', 'numberHighRating',
                    'numberMessageReceived', 'numberMessageRead', 'readAllMessage']


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