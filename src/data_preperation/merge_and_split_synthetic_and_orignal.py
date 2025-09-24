#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# In[3]:


path = r"C:\Users\User\PycharmProjects\master_thesis\simulation_data\final_run_data_preparation\data_splits_syn\full_preprocessed_100_patients.csv"
df_syntehtic = pd.read_csv(path)
path_original= r"C:\Users\User\PycharmProjects\master_thesis\simulation_data\final_run_data_preparation\data_splits\original_training_dataset.csv"
df_original = pd.read_csv(path_original)


# In[4]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the data
path = r"C:\Users\User\PycharmProjects\master_thesis\simulation_data\final_run_data_preparation\data_splits_syn\full_preprocessed_100_patients.csv"
df_synthetic = pd.read_csv(path)
path_original = r"C:\Users\User\PycharmProjects\master_thesis\simulation_data\final_run_data_preparation\data_splits\original_training_dataset.csv"
df_original = pd.read_csv(path_original)

print("Original columns:", df_original.columns.tolist())
print("Synthetic columns:", df_synthetic.columns.tolist())

def fix_timestamp_formats(df, is_synthetic=False):
    """Fix timestamp formats before merging"""
    df_copy = df.copy()

    # Ensure both timestamp columns exist
    timestamp_columns = ['timestamp', 'serverTimestamp']

    for col in timestamp_columns:
        if col in df_copy.columns:
            print(f"Fixing {col} column...")

            # Convert to string first to handle mixed types
            df_copy[col] = df_copy[col].astype(str)

            # Use pandas to_datetime with mixed format to handle different formats
            df_copy[col] = pd.to_datetime(df_copy[col], format='mixed', errors='coerce')

            # Check for any parsing failures
            null_count = df_copy[col].isnull().sum()
            if null_count > 0:
                print(f"  Warning: {null_count} timestamps in {col} could not be parsed")
        else:
            print(f"  Column {col} not found in dataframe")

    return df_copy

# Fix timestamps in both dataframes BEFORE merging
print("\nFixing synthetic data timestamps...")
df_synthetic_fixed = fix_timestamp_formats(df_synthetic, is_synthetic=True)

print("\nFixing original data timestamps...")
df_original_fixed = fix_timestamp_formats(df_original, is_synthetic=False)

# Now merge the fixed dataframes
print("\nMerging dataframes...")
df_merged = pd.concat([df_synthetic_fixed, df_original_fixed], ignore_index=True)

print(f"Merged dataframe shape: {df_merged.shape}")
print(f"Merged columns: {df_merged.columns.tolist()}")

def scale_features_fixed(df):
    """Scale features between 0-1 with MinMax scaler - with timestamp handling"""
    columns_NOT_to_scale = ['user_id', 'timestamp', 'serverTimestamp', 'action', 'reward', 'day_no']
    columns_to_scale = [col for col in df.columns if col not in columns_NOT_to_scale]

    print(f"Columns to scale: {columns_to_scale}")

    scaler = MinMaxScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df, scaler

def scale_features_with_scaler_fixed(df, scaler):
    """Scale features using a pre-fitted scaler - with timestamp handling"""
    columns_NOT_to_scale = ['user_id', 'timestamp', 'serverTimestamp', 'action', 'reward', 'day_no']
    columns_to_scale = [col for col in df.columns if col not in columns_NOT_to_scale]

    df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    return df

def generate_train_val_split_stratified_fixed(df, test_size=0.25, random_state=42):
    """
    Generate a stratified train/validation split that keeps entire patients together
    and maintains equal proportions of patient_ and ML IDs in both sets
    """

    # Create ID type for stratification
    unique_users = df['user_id'].unique()
    user_types = pd.DataFrame({
        'user_id': unique_users,
        'id_type': ['patient' if uid.startswith('patient_') else 'ML' for uid in unique_users]
    })

    print(f"Total users: {len(unique_users)}")
    print(f"Patient IDs: {sum(user_types['id_type'] == 'patient')}")
    print(f"ML IDs: {sum(user_types['id_type'] == 'ML')}")

    # Stratified split of user IDs
    train_users, val_users = train_test_split(
        user_types['user_id'],
        test_size=test_size,
        stratify=user_types['id_type'],
        random_state=random_state
    )

    # Create train and validation sets based on user IDs
    train_df = df[df['user_id'].isin(train_users)].reset_index(drop=True)
    val_df = df[df['user_id'].isin(val_users)].reset_index(drop=True)

    print(f"\nBefore scaling:")
    print(f"Train set: {len(train_df)} samples, {train_df['user_id'].nunique()} users")
    print(f"Val set: {len(val_df)} samples, {val_df['user_id'].nunique()} users")

    # Apply scaling with fixed timestamp handling
    train_df_scaled, scaler = scale_features_fixed(train_df.copy())
    val_df_scaled = scale_features_with_scaler_fixed(val_df.copy(), scaler)

    # Statistics for entries
    train_patient_entries = sum(train_df['user_id'].str.startswith('patient_'))
    train_ml_entries = sum(train_df['user_id'].str.startswith('ML'))
    val_patient_entries = sum(val_df['user_id'].str.startswith('patient_'))
    val_ml_entries = sum(val_df['user_id'].str.startswith('ML'))

    # Statistics for unique users
    train_patient_users = train_df[train_df['user_id'].str.startswith('patient_')]['user_id'].nunique()
    train_ml_users = train_df[train_df['user_id'].str.startswith('ML')]['user_id'].nunique()
    val_patient_users = val_df[val_df['user_id'].str.startswith('patient_')]['user_id'].nunique()
    val_ml_users = val_df[val_df['user_id'].str.startswith('ML')]['user_id'].nunique()

    print(f"\nAfter scaling:")
    print(f"Train set: {len(train_df)} samples, {train_df['user_id'].nunique()} users")
    print(f"  - Patient entries: {train_patient_entries} (from {train_patient_users} unique patients)")
    print(f"  - ML entries: {train_ml_entries} (from {train_ml_users} unique ML users)")

    print(f"Val set: {len(val_df)} samples, {val_df['user_id'].nunique()} users")
    print(f"  - Patient entries: {val_patient_entries} (from {val_patient_users} unique patients)")
    print(f"  - ML entries: {val_ml_entries} (from {val_ml_users} unique ML users)")

    return train_df, val_df, train_df_scaled, val_df_scaled, scaler

# Generate the train/val split with fixed timestamp handling
print("\nGenerating train/val split...")
train_df, val_df, train_df_scaled, val_df_scaled, scaler = generate_train_val_split_stratified_fixed(
    df_merged, test_size=0.25, random_state=42
)

# Save the results
import os
import pickle

save_path = r"C:\Users\User\PycharmProjects\master_thesis\simulation_data\final_run_data_preparation\data_splits_syn"

print(f"\nSaving files to: {save_path}")

# Save all versions
val_df.to_csv(os.path.join(save_path, "...csv"), index=False)
train_df.to_csv(os.path.join(save_path, "....csv"), index=False)
train_df_scaled.to_csv(os.path.join(save_path, "....csv"), index=False)
val_df_scaled.to_csv(os.path.join(save_path, "...csv"), index=False)

with open(os.path.join(save_path, "scaler_merged.pkl"), 'wb') as f:
    pickle.dump(scaler, f)

print("Files successfully saved!")

# Verify the timestamps are properly formatted
print(f"\nTimestamp verification:")
print(f"Train timestamp dtype: {train_df_scaled['timestamp'].dtype}")
print(f"Val timestamp dtype: {val_df_scaled['timestamp'].dtype}")
print("Sample timestamps from train:")
print(train_df_scaled['timestamp'].head())
print("Sample timestamps from val:")
print(val_df_scaled['timestamp'].head())


# In[6]:


# merge the two dataframes concatinating the two dataframes
df_merged = pd.concat([df_syntehtic, df_original], ignore_index=True)





# In[10]:


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


# In[11]:


from sklearn.model_selection import train_test_split
import pandas as pd

def generate_train_val_split_stratified(df, test_size=0.25, random_state=42):
    """
    Generate a stratified train/validation split that keeps entire patients together
    and maintains equal proportions of patient_ and ML IDs in both sets
    """

    # Erstelle ID-Typ für Stratifizierung
    unique_users = df['user_id'].unique()
    user_types = pd.DataFrame({
        'user_id': unique_users,
        'id_type': ['patient' if uid.startswith('patient_') else 'ML' for uid in unique_users]
    })

    print(f"Total users: {len(unique_users)}")
    print(f"Patient IDs: {sum(user_types['id_type'] == 'patient')}")
    print(f"ML IDs: {sum(user_types['id_type'] == 'ML')}")

    # Stratifizierte Aufteilung der User IDs
    train_users, val_users = train_test_split(
        user_types['user_id'],
        test_size=test_size,
        stratify=user_types['id_type'],
        random_state=random_state
    )

    # Erstelle Train- und Validierungssets basierend auf User IDs
    train_df = df[df['user_id'].isin(train_users)].reset_index(drop=True)
    val_df = df[df['user_id'].isin(val_users)].reset_index(drop=True)

    train_df_scaled, scaler = scale_features(train_df.copy())
    val_df_scaled = scale_features_with_scaler(val_df.copy(), scaler)

    # Statistiken für Einträge
    train_patient_entries = sum(train_df['user_id'].str.startswith('patient_'))
    train_ml_entries = sum(train_df['user_id'].str.startswith('ML'))
    val_patient_entries = sum(val_df['user_id'].str.startswith('patient_'))
    val_ml_entries = sum(val_df['user_id'].str.startswith('ML'))

    # Statistiken für eindeutige Benutzer
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

# Verwendung
train_df, val_df, train_df_scaled, val_df_scaled, scaler = generate_train_val_split_stratified(df_merged, test_size=0.25, random_state=42)


# In[13]:


import os
import pickle

# Definieren Sie den gewünschten Speicherort
save_path = r"C:\Users\User\PycharmProjects\master_thesis\simulation_data\final_run_data_preparation\data_splits_syn"

# Speichern mit vollständigen Pfaden
val_df.to_csv(os.path.join(save_path, "merged_syn_original_val.csv"), index=False)
train_df.to_csv(os.path.join(save_path, "merged_train_original_val.csv"), index=False)
train_df_scaled.to_csv(os.path.join(save_path, "merged_train_original_val_scaled.csv"), index=False)
val_df_scaled.to_csv(os.path.join(save_path, "merged_val_original_val_scaled.csv"), index=False)

with open(os.path.join(save_path, "scaler_merged.pkl"), 'wb') as f:
    pickle.dump(scaler, f)

print("Dateien erfolgreich gespeichert!")





# In[14]:


train_df


# In[28]:


def normalize_timestamps(df):
    """Normalisiert Zeitstempel auf einheitliches Format"""
    df_copy = df.copy()

    # Verwende pandas to_datetime mit mixed format
    try:
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], format='mixed', errors='coerce')
        print("Zeitstempel erfolgreich normalisiert mit 'mixed' format")
    except:
        # Fallback: ISO8601 format
        try:
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], format='ISO8601', errors='coerce')
            print("Zeitstempel erfolgreich normalisiert mit 'ISO8601' format")
        except:
            # Letzter Fallback: infer format
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], infer_datetime_format=True, errors='coerce')
            print("Zeitstempel normalisiert mit infer_datetime_format")

    # Überprüfe auf NaT (Not a Time) Werte
    nat_count = df_copy['timestamp'].isna().sum()
    if nat_count > 0:
        print(f"Warnung: {nat_count} Zeitstempel konnten nicht konvertiert werden")

    return df_copy

# Normalisiere die Zeitstempel
train_df_fixed_scaled = normalize_timestamps(train_df_scaled)
val_df_fixed_scaled = normalize_timestamps(train_df_scaled)


train_df_fixed = normalize_timestamps(train_df)
val_df_fixed = normalize_timestamps(val_df)


# In[29]:


# Überprüfung auf inkonsistente Zeitstempel-Formate
def check_timestamp_formats(df):
    """Überprüft verschiedene Zeitstempel-Formate im DataFrame"""

    # Konvertiere zu String falls nicht bereits
    timestamps = df['timestamp'].astype(str)

    # Finde verschiedene Formate
    formats = {
        'standard': [],  # YYYY-MM-DD HH:MM:SS
        'with_microseconds': [],  # YYYY-MM-DD HH:MM:SS.microseconds
        'date_only': [],  # YYYY-MM-DD
        'other': []
    }

    for i, ts in enumerate(timestamps):
        if pd.isna(ts) or ts == 'nan':
            continue

        # Standard Format: YYYY-MM-DD HH:MM:SS
        if len(ts) == 19 and ts.count(':') == 2 and ts.count('-') == 2:
            formats['standard'].append((i, ts))
        # Mit Mikrosekunden: YYYY-MM-DD HH:MM:SS.microseconds
        elif '.' in ts and ts.count(':') == 2 and ts.count('-') == 2:
            formats['with_microseconds'].append((i, ts))
        # Nur Datum: YYYY-MM-DD
        elif len(ts) == 10 and ts.count('-') == 2:
            formats['date_only'].append((i, ts))
        else:
            formats['other'].append((i, ts))

    # Zeige Statistiken
    print("Zeitstempel-Format Analyse:")
    print(f"Standard Format (YYYY-MM-DD HH:MM:SS): {len(formats['standard'])}")
    print(f"Mit Mikrosekunden: {len(formats['with_microseconds'])}")
    print(f"Nur Datum: {len(formats['date_only'])}")
    print(f"Andere Formate: {len(formats['other'])}")

    # Zeige Beispiele problematischer Formate
    if formats['with_microseconds']:
        print(f"\nBeispiele mit Mikrosekunden (erste 5):")
        for i, ts in formats['with_microseconds'][:5]:
            print(f"  Index {i}: {ts}")

    if formats['other']:
        print(f"\nAndere problematische Formate (erste 5):")
        for i, ts in formats['other'][:5]:
            print(f"  Index {i}: {ts}")

    return formats

# Überprüfe die Zeitstempel-Formate
 #train_df_scaled, val_df_scaled,
timestamp_formats = check_timestamp_formats(val_df_fixed)


# In[ ]:





# In[25]:


def normalize_timestamps(df):
    """Normalisiert Zeitstempel auf einheitliches Format"""
    df_copy = df.copy()

    # Verwende pandas to_datetime mit mixed format
    try:
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], format='mixed', errors='coerce')
        print("Zeitstempel erfolgreich normalisiert mit 'mixed' format")
    except:
        # Fallback: ISO8601 format
        try:
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], format='ISO8601', errors='coerce')
            print("Zeitstempel erfolgreich normalisiert mit 'ISO8601' format")
        except:
            # Letzter Fallback: infer format
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], infer_datetime_format=True, errors='coerce')
            print("Zeitstempel normalisiert mit infer_datetime_format")

    # Überprüfe auf NaT (Not a Time) Werte
    nat_count = df_copy['timestamp'].isna().sum()
    if nat_count > 0:
        print(f"Warnung: {nat_count} Zeitstempel konnten nicht konvertiert werden")

    return df_copy

# Normalisiere die Zeitstempel
train_df_fixed_scaled = normalize_timestamps(train_df_scaled)
val_df_fixed_scaled = normalize_timestamps(train_df_scaled)


train_df_fixed = normalize_timestamps(train_df)
val_df_fixed = normalize_timestamps(val_df)


# In[23]:


generate_train_val_split_stratified(df_merged, test_size=0.25, random_state=42)
import os
import pickle

# Definieren Sie den gewünschten Speicherort
save_path = r"C:\Users\User\PycharmProjects\master_thesis\simulation_data\final_run_data_preparation\data_splits_syn"

# Speichern mit vollständigen Pfaden
val_df_fixed.to_csv(os.path.join(save_path, "merged_syn_original_val.csv"), index=False)
train_df_fixed.to_csv(os.path.join(save_path, "merged_train_original_val.csv"), index=False)
train_df_fixed_scaled.to_csv(os.path.join(save_path, "merged_train_original_scaled.csv"), index=False)
val_df_fixed_scaled.to_csv(os.path.join(save_path, "merged_val_original_scaled.csv"), index=False)

with open(os.path.join(save_path, "scaler_merged.pkl"), 'wb') as f:
    pickle.dump(scaler, f)

print("Dateien erfolgreich gespeichert!")


# In[1]:


# investigage the full synthetic data


# In[2]:


import pandas as pd


# In[7]:


path_train = r"C:\Users\User\PycharmProjects\master_thesis\simulation_data\final_run_data_preparation\data_splits_syn\full_preprocessed_100_patients.csv"

# load the dataset 
df_train = pd.read_csv(path_train)


# In[8]:


len(df_train["user_id"].unique())


# In[42]:


# load synthetic data
path_syn = r"C:\Users\User\PycharmProjects\master_thesis\simulation_data\final_run_data_preparation\data_splits_syn\full_preprocessed_100_patients.csv"


path_syn = r"C:\Users\User\PycharmProjects\master_thesis\simulation_data\final_run_data_preparation\data_splits_syn\test_df.csv"


# In[43]:


# load the data = 
df_syn = pd.read_csv(path_syn)


# In[44]:


len(df_syn["user_id"].unique())


# In[ ]:




