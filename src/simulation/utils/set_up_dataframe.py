import pandas as pd
import os
def init_custom_df():
    """Intialised the dataframe with the correct columns"""
    cols = [
        'day_no', 'day_part_x', 'user_id', 'actual_rating', 'numberRating',
        'highestRating', 'lowestRating', 'medianRating', 'sdRating',
        'numberLowRating', 'numberMediumRating', 'numberHighRating',
        'numberMessageReceived', 'numberMessageRead', 'readAllMessage',
        'reward', 'rl_action', 'motivation_preference'
    ]
    return pd.DataFrame(columns=cols)

def save_trajectories_to_csv_data(trajectories, filename):
    """
    saves the trajctories in a dataframe to a csv file
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    trajectories.to_csv(filename, index=False)

    return trajectories
