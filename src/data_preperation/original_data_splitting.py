from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
from simulation_data.final_run_data_preparation import preprocessing


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
        Create train/validation/test splits using GroupShuffleSplit to ensure no user overlap

        Args:
            df: Input dataframe
            test_size: Proportion for test set (from total data)
            val_size: Proportion for validation set (from remaining train pool)
            random_state_test: Random state for train/test split
            random_state_val: Random state for train/validation split

        Returns:
            Dictionary containing all datasets and metadata
        """
        unique_users = df['user_id'].unique()
        print(f"Total unique users: {len(unique_users)}")

        # Step 1: Create train_pool and test split
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state_test)
        train_pool_indices, test_indices = next(splitter.split(df, groups=df['user_id']))

        train_pool_df = df.iloc[train_pool_indices].copy()
        test_df = df.iloc[test_indices].copy()

        # Keep original training dataset (before val split)
        original_training_dataset = train_pool_df.copy()

        # Step 2: Split train_pool into train and validation
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
        print("✓ No user overlap between splits confirmed")

        # Scale features: fit on train, transform val and test
        train_df_scaled, scaler = self.scale_features(train_df)
        val_df_scaled = self.scale_features_with_existing_scaler(val_df, scaler)
        test_df_scaled = self.scale_features_with_existing_scaler(test_df, scaler)

        # Store scaler for later use
        self.scaler = scaler

        # Create results dictionary
        results = {
            # Original (unscaled) datasets
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'original_training_dataset': original_training_dataset,  # Before val split

            # Scaled datasets
            'train_df_scaled': train_df_scaled,
            'val_df_scaled': val_df_scaled,
            'test_df_scaled': test_df_scaled,

            # Scaler and metadata
            'scaler': scaler,
            'split_info': {
                'train_users': sorted(train_user_ids),
                'val_users': sorted(val_user_ids),
                'test_users': sorted(test_user_ids),
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'test_size': test_size,
                'val_size': val_size,
                'random_state_test': random_state_test,
                'random_state_val': random_state_val
            }
        }

        return results

    def save_all_datasets(self, results, output_dir='data_splits', timestamp_suffix=True):
        """
        Save all datasets, scaler, and metadata

        Args:
            results: Dictionary from create_train_val_test_split
            output_dir: Directory to save files
            timestamp_suffix: Whether to add timestamp to filenames
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

        with open(f'{output_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(results['scaler'], f)

        split_info = results['split_info']
        with open(f'{output_dir}/split_info.txt', 'w') as f:
            f.write("DATA SPLIT INFORMATION\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Split Parameters:\n")
            f.write(f"  Test size: {split_info['test_size']}\n")
            f.write(f"  Validation size: {split_info['val_size']}\n")
            f.write(f"  Random state (test): {split_info['random_state_test']}\n")
            f.write(f"  Random state (val): {split_info['random_state_val']}\n\n")

            f.write(f"Dataset Sizes:\n")
            f.write(f"  Training: {split_info['train_samples']} samples\n")
            f.write(f"  Validation: {split_info['val_samples']} samples\n")
            f.write(f"  Test: {split_info['test_samples']} samples\n\n")

            f.write(f"User Distribution:\n")
            f.write(f"  Training users ({len(split_info['train_users'])}): {split_info['train_users']}\n")
            f.write(f"  Validation users ({len(split_info['val_users'])}): {split_info['val_users']}\n")
            f.write(f"  Test users ({len(split_info['test_users'])}): {split_info['test_users']}\n\n")

            f.write(f"Files Generated:\n")
            f.write(f"  - train_df.csv (original training set)\n")
            f.write(f"  - val_df.csv (original validation set)\n")
            f.write(f"  - test_df.csv (original test set)\n")
            f.write(f"  - original_training_dataset.csv (before val split)\n")
            f.write(f"  - train_df_scaled.csv (scaled training set)\n")
            f.write(f"  - val_df_scaled.csv (scaled validation set)\n")
            f.write(f"  - test_df_scaled.csv (scaled test set)\n")
            f.write(f"  - scaler.pkl (MinMaxScaler object)\n")
            f.write(f"  - split_info.txt (this file)\n")

        print(f"\n✓ All datasets saved to '{output_dir}/' directory")
        return f'{output_dir}',

    def load_scaler(self, scaler_path):
        """Load a previously saved scaler"""
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        self.scaler = scaler
        return scaler


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

