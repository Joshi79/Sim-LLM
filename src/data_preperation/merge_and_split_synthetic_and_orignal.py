
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import pickle

def normalize_timestamps(df):
    """Fix errors in downstream RL pipline due to inconsistent timestamp formats"""
    df_copy = df.copy()

    try:
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], format='mixed', errors='coerce')
    except:
        try:
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], format='ISO8601', errors='coerce')
        except:
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], infer_datetime_format=True, errors='coerce')

    nat_count = df_copy['timestamp'].isna().sum()
    if nat_count > 0:
        print(f"Warning: {nat_count} couldnt convert the timestamp")

    return df_copy

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


def generate_train_val_split_stratified(df, test_size=0.25, random_state=42):
    """
    Generate a stratified train/validation split that keeps entire patients together
    and maintains equal proportions of patient_ and ML IDs in both sets
    """

    unique_users = df['user_id'].unique()
    user_types = pd.DataFrame({
        'user_id': unique_users,
        'id_type': ['patient' if uid.startswith('patient_') else 'ML' for uid in unique_users]
    })

    print(f"Total users: {len(unique_users)}")
    print(f"Patient IDs: {sum(user_types['id_type'] == 'patient')}")
    print(f"ML IDs: {sum(user_types['id_type'] == 'ML')}")

    train_users, val_users = train_test_split(
        user_types['user_id'],
        test_size=test_size,
        stratify=user_types['id_type'],
        random_state=random_state
    )

    train_df = df[df['user_id'].isin(train_users)].reset_index(drop=True)
    val_df = df[df['user_id'].isin(val_users)].reset_index(drop=True)

    train_df_scaled, scaler = scale_features(train_df.copy())
    val_df_scaled = scale_features_with_scaler(val_df.copy(), scaler)

    train_patient_entries = sum(train_df['user_id'].str.startswith('patient_'))
    train_ml_entries = sum(train_df['user_id'].str.startswith('ML'))
    val_patient_entries = sum(val_df['user_id'].str.startswith('patient_'))
    val_ml_entries = sum(val_df['user_id'].str.startswith('ML'))

    train_patient_users = train_df[train_df['user_id'].str.startswith('patient_')]['user_id'].nunique()
    train_ml_users = train_df[train_df['user_id'].str.startswith('ML')]['user_id'].nunique()
    val_patient_users = val_df[val_df['user_id'].str.startswith('patient_')]['user_id'].nunique()
    val_ml_users = val_df[val_df['user_id'].str.startswith('ML')]['user_id'].nunique()

    print(f"\nTrain set: {len(train_df)} samples, {train_df['user_id'].nunique()} users")
    print(f"  - Patient entries: {train_patient_entries} (from {train_patient_users} unique patients)")
    print(f"  - ML entries: {train_ml_entries} (from {train_ml_users} unique ML users)")

    print(f"Val set: {len(val_df)} samples, {val_df['user_id'].nunique()} users")
    print(f"  - Patient entries: {val_patient_entries} (from {val_patient_users} unique patients)")
    print(f"  - ML entries: {val_ml_entries} (from {val_ml_users} unique ML users)")

    return train_df, val_df, train_df_scaled, val_df_scaled, scaler





print('Load Data')
path = r"../../data/synthetic_data_baseline_prompt.csv"
df_synthetic = pd.read_csv(path)
path_original= r"../../data/original_training_dataset.csv"
df_original = pd.read_csv(path_original)

print("Original columns:", df_original.columns.tolist())
print("Synthetic columns:", df_synthetic.columns.tolist())

df_merged = pd.concat([df_synthetic, df_original], ignore_index=True)

train_df, val_df, train_df_scaled, val_df_scaled, scaler = generate_train_val_split_stratified(df_merged,test_size=0.25,random_state=42)
train_df_fixed_scaled = normalize_timestamps(train_df_scaled)
val_df_fixed_scaled = normalize_timestamps(val_df_scaled)

train_df_fixed = normalize_timestamps(train_df)
val_df_fixed = normalize_timestamps(val_df)


save_path = r"../../data/"

val_df_fixed.to_csv(os.path.join(save_path, "merged_syn_original_val.csv"), index=False)
train_df_fixed.to_csv(os.path.join(save_path, "merged_train_original_val.csv"), index=False)
train_df_fixed_scaled.to_csv(os.path.join(save_path, "merged_train_original_scaled.csv"), index=False)
val_df_fixed_scaled.to_csv(os.path.join(save_path, "merged_val_original_scaled.csv"), index=False)

with open(os.path.join(save_path, "scaler_merged.pkl"), 'wb') as f:
    pickle.dump(scaler, f)





