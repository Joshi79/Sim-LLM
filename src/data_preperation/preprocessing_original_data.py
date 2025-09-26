import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit 
import numpy as np


# This code was provided by Rutger

seed = 1
np.random.seed(seed)


# Order based on timestamp and user_id, reset index
def order_df(df):
    # Reset the index
    df = df.reset_index(drop=True, inplace=False)


    # Transform timestamp into correct type and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['serverTimestamp'] = pd.to_datetime(df['serverTimestamp'])

    df = df.sort_values(['user_id','serverTimestamp', 'day_part_x'])

    # Reset the index
    df = df.reset_index(drop=True, inplace=False)

    return df




# Def for dropping redundant columns
def get_df_dropped_redundant_columns(df):
    return df[["serverTimestamp",
                "day_part_x",
                "user_id",
                "numberRating",
                "highestRating",
                "lowestRating",
                "medianRating",
                "sdRating",
                "numberLowRating",
                "numberMediumRating",
                "numberHighRating",
                "numberMessageReceived",
                "numberMessageRead",
                "readAllMessage",
                "reward",
                "timestamp",
                "action"]]

# Def for dropping erroneous rows. Namely numberRating, numberMessageReceived, numberMessageRead can be 1, 2, or 3 if day_part_x is 1, 2, or 3
# Keep one of the rows where serverTimestamp, day_part_x and user_id are equal
def remove_incorrect_rows(df):
    
    # Sort the DataFrame by the key columns and 'numberRating' in descending order
    print("Len df before removing duplicates on columns serverTimestamp, day_part_x, user_id, numberRating, numberMessageReceived", len(df))
    df_sorted = df.sort_values(by=['serverTimestamp', 'day_part_x', 'user_id', 'numberRating', 'numberMessageReceived'], ascending=[True, True, True, False, True])

    # Drop duplicates, keeping the first occurrence which is the row with the highest numberRating
    df = df_sorted.drop_duplicates(subset=['serverTimestamp', 'day_part_x', 'user_id'], keep='first')
    print("Len df after removing duplicates on columns serverTimestamp, day_part_x, user_id, numberRating ", len(df))
    return df





def drop_users_no_ratingsAndMessages(df):
    print("Start dropping users with no ratings and no messages read.")

    df_noRead_noRatings = df.groupby('user_id').filter(lambda x: (x['numberMessageRead'] == 0).all() and (x['numberRating'] == 0).all())

    # Get the user_ids
    user_ids = df_noRead_noRatings['user_id'].unique()
    print(user_ids)
    print("Number of users to remove (have no ratings and messages read).", len(user_ids))

    print("Length df before removing users with no ratings and messages read.", len(df))
    # Filter df
    df = df[~df['user_id'].isin(user_ids)]
    print("Length df after removing users with no ratings and messages read.", len(df))
    print("Nr of unique users after removing users with no messages read and no ratings:", df['user_id'].nunique())
    df = df.reset_index(drop=True, inplace=False)
    return df

# Def for dropping nan rows
def drop_nan_rows(df):
    df = df.dropna(inplace=False)
    df = df.reset_index(drop=True, inplace=False)
    return df

# Def for removing rows at end of trial where participants isn't active anymore. Not needed at beginning, can't make assumption whether they ignorder notifications, and almost from start activity shown in plots
def remove_redundant_rows_EndOfTrial(df):
    print("Start removing redundant rows.")

    # Transform timestamp into correct type and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['serverTimestamp'] = pd.to_datetime(df['serverTimestamp'])

    df = df.sort_values(['user_id','serverTimestamp', 'day_part_x'])
    # Reset the index
    df = df.reset_index(drop=True, inplace=False)

    print("Length df before filtering on redundant rows", len(df))

    # List filtered df's
    list_df_users = []

    # Group by 'user_id'
    grouped = df.groupby('user_id')

    # Loop over the groups
    for user_id, user_df in grouped:
        # Reset index
        user_df = user_df.reset_index(drop=True, inplace=False)
        print("Len df before filtering" + str(user_id), len(user_df))
        
        # Find first non-zero value backwards
        # Reverse the DataFrame
        user_df = user_df[::-1]

        # Reset index
        user_df = user_df.reset_index(drop=True, inplace=False)

        # Find the index of the first non-zero value in each column
        last_nonzero_index = user_df[(user_df['numberMessageRead'] > 0) | (user_df['numberRating'] > 0)].index[0]

        # Filter the df
        user_df = user_df[last_nonzero_index:]

        # Reset index
        user_df = user_df.reset_index(drop=True, inplace=False)
        print("Len resulting df " + str(user_id), len(user_df))

        list_df_users.append(user_df)

    # Concat user df's and order
    df = pd.concat(list_df_users)
    df = df.reset_index(drop=True, inplace=False)
    df = order_df(df)
    print("Length df after filtering on redundant rows", len(df))

    return df




# Scale features between 0-1 with MinMax scaler
def scale_features(df):

    # Columns to scale excluding some columns like user_id
    columns_NOT_to_scale = ['user_id', 'timestamp', 'serverTimestamp', 'action', 'reward']
    columns_to_scale = [col for col in df.columns if col not in columns_NOT_to_scale]
    # Scale the selected columns
    scaler = MinMaxScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df, scaler

# For scaling features in test set based on derived scaler from train set
def scale_features_test_set(df, scaler):
    # Columns to scale excluding some columns like user_id
    columns_NOT_to_scale = ['user_id', 'timestamp', 'serverTimestamp', 'action', 'reward']
    columns_to_scale = [col for col in df.columns if col not in columns_NOT_to_scale]
    # Scale the selected columns
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    return df




def perform_data_cleaning(df):
    

    df = get_df_dropped_redundant_columns(df)
    print("Redundant columns have been dropped.")
    print()


    df = remove_incorrect_rows(df)
    print("Erroneous rows have been removed")
    print()
    

    df = drop_users_no_ratingsAndMessages(df)
    print("Users with no ratings and messages removed")
    print()

    df = remove_redundant_rows_EndOfTrial(df)
    print("Redundant rows have been removed")
    print()

    print(df[df.isna().any(axis=1)])
    print("Number nan rows", len(df[df.isna().any(axis=1)]))

    df = order_df(df)

    return df




def train_val_test_generation(df):


    list_train_dfs = []
    list_val_dfs = []
    list_test_df = []


    # List for scaler
    list_scalers = []

    # Placeholder lists containing train_val df's and test df's, all unscaled
    list_train_val_dfs = []
    list_test_unscaled_dfs = []

    splitter = GroupShuffleSplit(test_size=.25, n_splits=20, random_state = 0) # was 0.2 en 0.25 voor train/val
    
    for train_val_inds, test_inds in splitter.split(df, groups=df['user_id']):
        # Get train_val and test and reset index
        train_val_df = df.iloc[train_val_inds]
        test_df = df.iloc[test_inds]
        train_val_df = train_val_df.reset_index(drop=True, inplace=False)
        test_df = test_df.reset_index(drop=True, inplace=False)

        # Add to corresponding lists
        list_train_val_dfs.append(train_val_df)
        list_test_unscaled_dfs.append(test_df)

    
    # Loop over 20 splits to do train-val split
    for i, train_val_df in enumerate(list_train_val_dfs):
        # Split
        splitter = GroupShuffleSplit(test_size=0.33333333, n_splits=1, random_state = 0)
        # Train and val index
        train_inds, val_inds = next(splitter.split(train_val_df, groups=train_val_df['user_id']))

        # Get train and val df
        train_df = train_val_df.iloc[train_inds]
        val_df = train_val_df.iloc[val_inds]
        train_df = train_df.reset_index(drop=True, inplace=False)
        val_df = val_df.reset_index(drop=True, inplace=False)


        # Perform scaling
        # Get scaler from train_df
        train_df_scaled, scaler = scale_features(train_df)
        # Scale features test df based on scaler from train_df
        val_df_scaled = scale_features_test_set(val_df, scaler)
        test_df_scaled = scale_features_test_set(list_test_unscaled_dfs[i], scaler)

        # Add scaled df's to corresponding lists
        list_train_dfs.append(train_df_scaled)
        list_val_dfs.append(val_df_scaled)
        list_test_df.append(test_df_scaled)

        # Add scaler to list
        list_scalers.append(scaler)


    return list_train_dfs, list_val_dfs, list_test_df, list_scalers