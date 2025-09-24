"""
This makes the synthetic data compatible with the offline RL algorithmus used in the project.
"""

import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit


def create_synthetic_timestamp(df):
    """Create synthetic timestamp and serverTimestamp columns from day_no and day_part_x"""
    df['timestamp'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(df['day_no'], unit='D') + pd.to_timedelta(
        df['day_part_x'] * 8, unit='h')
    df['serverTimestamp'] = df['timestamp']
    return df


def parse_actual_rating(df):
    """Parse the actual_rating column that contains string representations of lists"""

    def safe_parse_rating(rating_str):
        try:
            if pd.isna(rating_str) or rating_str == 'None':
                return []
            if isinstance(rating_str, str):
                if rating_str.startswith('[') and rating_str.endswith(']'):
                    return ast.literal_eval(rating_str)
                else:
                    return [int(rating_str.strip('[]'))]
            else:
                return [rating_str] if not pd.isna(rating_str) else []
        except:
            return []

    df['actual_rating_parsed'] = df['actual_rating'].apply(safe_parse_rating)
    return df


def order_df_syn(df):
    """Order based on user_id, day_no and day_part_x, reset index"""
    df = df.reset_index(drop=True, inplace=False)
    df = df.sort_values(['user_id', 'day_no', 'day_part_x'])
    df = df.reset_index(drop=True, inplace=False)
    return df


def rename_columns_for_compatibility(df):
    """Rename rl_action to action to match the expected format"""
    df = df.rename(columns={'rl_action': 'action'})
    return df


def get_df_dropped_redundant_columns_syn(df):
    """Select relevant columns for RL algorithm"""
    return df[["serverTimestamp",
               "day_part_x",
               "day_no",
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


def remove_incorrect_rows_syn(df):
    """Keep one of the rows where day_no, day_part_x and user_id are equal (if any duplicates)"""
    print("Len df before removing duplicates on columns day_no, day_part_x, user_id:", len(df))

    df_sorted = df.sort_values(by=['day_no', 'day_part_x', 'user_id', 'numberRating'],
                               ascending=[True, True, True, False])

    df = df_sorted.drop_duplicates(subset=['day_no', 'day_part_x', 'user_id'], keep='first')
    print("Len df after removing duplicates on columns day_no, day_part_x, user_id:", len(df))
    return df


def drop_users_no_ratingsAndMessages_syn(df):
    """Drop users with no ratings inputted and no messages read"""
    print("Start dropping users with no ratings and no messages read.")

    df_noRead_noRatings = df.groupby('user_id').filter(
        lambda x: (x['numberMessageRead'] == 0).all() and (x['numberRating'] == 0).all()
    )

    user_ids = df_noRead_noRatings['user_id'].unique()
    print("User IDs to remove:", user_ids)
    print("Number of users to remove (have no ratings and messages read):", len(user_ids))

    print("Length df before removing users with no ratings and messages read:", len(df))
    df = df[~df['user_id'].isin(user_ids)]
    print("Length df after removing users with no ratings and messages read:", len(df))
    print("Nr of unique users after removing users with no messages read and no ratings:", df['user_id'].nunique())
    df = df.reset_index(drop=True, inplace=False)
    return df


def drop_nan_rows_syn(df):
    """Drop NaN rows"""
    print("Dropping NaN rows...")
    print("Length before dropping NaN:", len(df))
    df = df.dropna(inplace=False)
    df = df.reset_index(drop=True, inplace=False)
    print("Length after dropping NaN:", len(df))
    return df


def remove_redundant_rows_EndOfTrial_syn(df):
    """Remove rows at end of trial where participants aren't active anymore"""
    print("Start removing redundant rows at end of trial.")

    df = df.sort_values(['user_id', 'day_no', 'day_part_x'])
    df = df.reset_index(drop=True, inplace=False)

    print("Length df before filtering on redundant rows:", len(df))

    list_df_users = []
    grouped = df.groupby('user_id')

    for user_id, user_df in grouped:
        user_df = user_df.reset_index(drop=True, inplace=False)

        user_df_reversed = user_df[::-1].reset_index(drop=True, inplace=False)
        active_mask = (user_df_reversed['numberMessageRead'] > 0) | (user_df_reversed['numberRating'] > 0)

        if active_mask.any():
            last_active_index = active_mask.idxmax()
            user_df_filtered = user_df_reversed[last_active_index:].iloc[::-1].reset_index(drop=True, inplace=False)
        else:
            user_df_filtered = user_df

        list_df_users.append(user_df_filtered)

    df = pd.concat(list_df_users, ignore_index=True)
    df = order_df_syn(df)
    print("Length df after filtering on redundant rows:", len(df))

    return df


def calculate_ph_rl_reward(df):
    """
    Calculate pH-RL reward based on the methodology:
    reward = 0.5 * (messages_read/messages_received) + 0.5 * (number_ratings)
    """
    df_copy = df.copy()

    required_cols = ['numberMessageReceived', 'numberMessageRead', 'numberRating']
    for col in required_cols:
        if col not in df_copy.columns:
            raise ValueError(f"Column '{col}' is missing from the DataFrame.")

    df_copy['reward'] = 0.0

    for row in df_copy.itertuples():
        if row.numberMessageReceived > 0:
            message_read_fraction = row.numberMessageRead / row.numberMessageReceived
        else:
            message_read_fraction = 0.0

        reward = 0.5 * message_read_fraction + 0.5 * row.numberRating
        df_copy.at[row.Index, 'reward'] = round(reward, 6)

    return df_copy


def perform_data_cleaning(df):
    """Main function to perform all data cleaning steps for synthetic data"""

    print("Starting data cleaning for synthetic data...")
    print(f"Initial data shape: {df.shape}")
    print(f"Initial columns: {list(df.columns)}")

    # Parse actual_rating column
    df = parse_actual_rating(df)
    print("Actual rating column parsed")

    # Create synthetic timestamps
    df = create_synthetic_timestamp(df)
    print("Synthetic timestamps created")

    # Rename columns for compatibility
    df = rename_columns_for_compatibility(df)
    print("Columns renamed for compatibility")

    # Select relevant columns
    df = get_df_dropped_redundant_columns_syn(df)
    print("Redundant columns dropped\n")

    # Remove incorrect/duplicate rows
    df = remove_incorrect_rows_syn(df)
    print("Duplicate rows removed\n")

    # Drop users with no activity
    df = drop_users_no_ratingsAndMessages_syn(df)
    print("Inactive users removed\n")

    # Remove redundant rows at end of trial
    df = remove_redundant_rows_EndOfTrial_syn(df)
    print("End-of-trial redundant rows removed\n")

    # Check for NaN values
    nan_rows = df[df.isna().any(axis=1)]
    print(f"Number of NaN rows: {len(nan_rows)}")
    if len(nan_rows) > 0:
        print("NaN rows found - removing...")
        df = drop_nan_rows_syn(df)

    # Final ordering
    df = order_df_syn(df)

    print(f"\nData cleaning complete!")
    print(f"Final cleaned data shape: {df.shape}")
    print(f"Number of unique users: {df['user_id'].nunique()}")

    return df


def scale_features(df):
    """Scale features between 0-1 with MinMax scaler"""
    columns_NOT_to_scale = ['user_id', 'timestamp', 'serverTimestamp', 'action', 'reward', 'day_no']
    columns_to_scale = [col for col in df.columns if col not in columns_NOT_to_scale]

    print(f"Columns to scale: {columns_to_scale}")

    scaler = MinMaxScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df, scaler


def scale_features_with_scaler(df, scaler):
    """Scale features using a pre-fitted scaler"""
    columns_NOT_to_scale = ['user_id', 'timestamp', 'serverTimestamp', 'action', 'reward', 'day_no']
    columns_to_scale = [col for col in df.columns if col not in columns_NOT_to_scale]

    df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    return df


def generate_train_val_split(df, test_size=0.25, random_state=42):
    """
    Generate a single train/validation split

    Args:
        df: Cleaned dataframe
        test_size: Proportion of data for validation (default 0.3 for 70/30 split)
        random_state: Random seed for reproducibility

    Returns:
        train_df: Training dataframe (scaled)
        val_df: Validation dataframe (scaled)
        scaler: Fitted MinMaxScaler from training data
    """

    print(f"\nGenerating train/validation split (train: {(1 - test_size) * 100:.0f}%, val: {test_size * 100:.0f}%)...")

    # Create train/val split using GroupShuffleSplit
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)

    # Get the split indices
    train_inds, val_inds = next(splitter.split(df, groups=df['user_id']))

    # Create train and validation sets
    train_df = df.iloc[train_inds].reset_index(drop=True)
    val_df = df.iloc[val_inds].reset_index(drop=True)

    print(f"Train set: {len(train_df)} samples, {train_df['user_id'].nunique()} users")
    print(f"Val set: {len(val_df)} samples, {val_df['user_id'].nunique()} users")

    # Scale features
    print("\nScaling features...")
    train_df_scaled, scaler = scale_features(train_df.copy())
    val_df_scaled = scale_features_with_scaler(val_df.copy(), scaler)

    print("✓ Scaling complete")

    return train_df_scaled, val_df_scaled, train_df, val_df, scaler