import pandas as pd
import numpy as np
import os
import pickle
from preprocessing_synthetic_data import perform_data_cleaning, generate_train_val_split



np.random.seed(42)


def main():

    path_synthetic = r"../../data/synthetic_data_baseline_prompt.csv"
    synthetic_df = pd.read_csv(path_synthetic)
    print(f"  Shape: {synthetic_df.shape}")
    print(f"  Unique users: {synthetic_df['user_id'].nunique()}")

    # this ensures that the format is the same as Rutgers format - but no patients are removed
    synthetic_df_cleaned = perform_data_cleaning(synthetic_df.copy())

    # create subset of 60 patients
    unique_users = synthetic_df_cleaned['user_id'].unique()
    selected_users = unique_users[:60]
    synthetic_60 = synthetic_df_cleaned[synthetic_df_cleaned['user_id'].isin(selected_users)].reset_index(drop=True)

    train_df_scaled, val_df_scaled, train_df, val_df, scaler = generate_train_val_split(synthetic_df_cleaned,test_size=0.25,random_state=42)
    train_60_scaled, val_60_scaled, train_60, val_60, scaler_60 = generate_train_val_split(synthetic_60,test_size=0.25,random_state=42)

    # Save the datasets
    output_dir = r'../../data/synthetic_data_splits'
    os.makedirs(output_dir, exist_ok=True)
    train_df_scaled.to_csv(f'{output_dir}/full_train_df_scaled_.csv', index=False)
    val_df_scaled.to_csv(f'{output_dir}/full_val_df_not_scaled.csv', index=False)
    train_df.to_csv(f'{output_dir}/synthetic_full_train_df.csv', index=False)
    val_df.to_csv(f'{output_dir}/synthetic_full_val_df.csv', index=False)

    train_60_scaled.to_csv(f'{output_dir}/60_train_df_scaled_.csv', index=False)
    val_60_scaled.to_csv(f'{output_dir}/60_val_df_scaled.csv', index=False)
    train_60.to_csv(f'{output_dir}/train_60.csv', index=False)
    val_60.to_csv(f'{output_dir}/val_60.csv', index=False)










if __name__ == "__main__":
    main()