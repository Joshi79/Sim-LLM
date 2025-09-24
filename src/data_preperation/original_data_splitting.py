from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
from src.data_preperation import preprocessing_original_data as preprocessing


class DataSplitter:
    def __init__(self):
        self.scaler = None

    def scale_features(self, df):
        """
        Scale features using MinMaxScaler, excluding certain columns
        """
        # Columns to exclude from scaling
        columns_NOT_to_scale = ['user_id', 'timestamp', 'serverTimestamp', 'action', 'reward']
        columns_to_scale = [col for col in df.columns if col not in columns_NOT_to_scale]

        # Scale the selected columns
        scaler = MinMaxScaler()
        df_scaled = df.copy()
        df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

        return df_scaled, scaler

    def scale_features_with_existing_scaler(self, df, scaler):
        """
        Scale features using an existing fitted scaler
        """
        columns_NOT_to_scale = ['user_id', 'timestamp', 'serverTimestamp', 'action', 'reward']
        columns_to_scale = [col for col in df.columns if col not in columns_NOT_to_scale]

        df_scaled = df.copy()
        df_scaled[columns_to_scale] = scaler.transform(df[columns_to_scale])
        return df_scaled

    def create_train_val_test_split(self, df, test_size=0.25, val_size=0.33, random_state_test=1, random_state_val=2):
        """
       Split the Data
        """
        unique_users = df['user_id'].unique()
        print(f"Total unique users: {len(unique_users)}")

        # Step 1: Create train_pool and test split
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state_test)
        train_pool_indices, test_indices = next(splitter.split(df, groups=df['user_id']))

        train_pool_df = df.iloc[train_pool_indices].copy()
        test_df = df.iloc[test_indices].copy()

        # Keep original training dataset for the LLM
        original_training_dataset = train_pool_df.copy()

        # Make a second split for training the DDQN - split the orignal data of 21 into training 14 and val 7
        val_splitter = GroupShuffleSplit(test_size=val_size, n_splits=1, random_state=random_state_val)
        train_indices, val_indices = next(val_splitter.split(train_pool_df, groups=train_pool_df['user_id']))

        train_df = train_pool_df.iloc[train_indices].copy()
        val_df = train_pool_df.iloc[val_indices].copy()

        # Get user statistics
        train_user_ids = train_df['user_id'].unique()
        val_user_ids = val_df['user_id'].unique()
        test_user_ids = test_df['user_id'].unique()

        print(f"Training Set: {len(train_df)} samples, {len(train_user_ids)} users")
        print(f"Validation Set: {len(val_df)} samples, {len(val_user_ids)} users")
        print(f"Test Set: {len(test_df)} samples, {len(test_user_ids)} users")
        print(f"Test users: {sorted(test_user_ids)}")

        # Verify no user overlap
        assert len(set(train_user_ids) & set(val_user_ids)) == 0, "Train-Val user overlap detected!"
        assert len(set(train_user_ids) & set(test_user_ids)) == 0, "Train-Test user overlap detected!"
        assert len(set(val_user_ids) & set(test_user_ids)) == 0, "Val-Test user overlap detected!"

        # Scale the training 14 and validation 7 set
        train_df_scaled, scaler_train = self.scale_features(train_df)
        val_df_scaled = self.scale_features_with_existing_scaler(val_df, scaler_train)

        # Scale the test set with the  original training dataset of 21
        _, scaler_original = self.scale_features(original_training_dataset)
        test_df_scaled = self.scale_features_with_existing_scaler(test_df, scaler_original)

        results = {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'original_training_dataset': original_training_dataset,

            # Scaled datasets
            'train_df_scaled': train_df_scaled,
            'val_df_scaled': val_df_scaled,
            'test_df_scaled': test_df_scaled
        }
        return results

    def save_all_datasets(self, results, output_dir='data_splits', timestamp_suffix=True):
        """
        Save the datasets
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)


        results['train_df'].to_csv(f'{output_dir}/train_df.csv', index=False)
        results['val_df'].to_csv(f'{output_dir}/val_df.csv', index=False)
        results['test_df'].to_csv(f'{output_dir}/test_df.csv', index=False)
        results['original_training_dataset'].to_csv(f'{output_dir}/original_training_dataset.csv', index=False)

        results['train_df_scaled'].to_csv(f'{output_dir}/train_df_scaled.csv', index=False)
        results['val_df_scaled'].to_csv(f'{output_dir}/val_df_scaled.csv', index=False)
        results['test_df_scaled'].to_csv(f'{output_dir}/test_df_scaled.csv', index=False)

        return f'{output_dir}',

def main():

    splitter = DataSplitter()
    df = pd.read_csv('../../data/merged_file')

    dataset_original_cleaned = preprocessing.perform_data_cleaning(df)

    print("Performing train/validation/test split...")
    results = splitter.create_train_val_test_split(dataset_original_cleaned, test_size=0.25, val_size=0.33)

    print("Saving all datasets and metadata...")
    output_dir = splitter.save_all_datasets(results, output_dir='../../data/data_splits')

    print(f"\n Files saved in '{output_dir}/' directory")

if __name__ == "__main__":
    main()

