import pandas as pd
import numpy as np
import os
import pickle
from preprocessing_syntehtic_data import (
    perform_data_cleaning,
    calculate_ph_rl_reward,
    generate_train_val_split
)

# Set seed for reproducibility
np.random.seed(42)


def main():

    path_synthetic = r"../../data/synthetic_data_baseline_prompt.csv"
    synthetic_df = pd.read_csv(path_synthetic)
    print(f"  Shape: {synthetic_df.shape}")
    print(f"  Unique users: {synthetic_df['user_id'].nunique()}")

    synthetic_df_cleaned = perform_data_cleaning(synthetic_df.copy())


    synthetic_df_cleaned = calculate_ph_rl_reward(synthetic_df_cleaned.copy())


    # Generate single train/validation split (70/30)
    print("\n4. GENERATING TRAIN/VALIDATION SPLIT")
    train_df_scaled, val_df_scaled, train_df, val_df, scaler = generate_train_val_split(
        synthetic_df_cleaned,
        test_size=0.25,
        random_state=42
    )

    # Create directory for saving
    print("\n5. SAVING DATA")
    print("-" * 40)

    output_dir = 'data_splits_syn'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created directory: {output_dir}/")

    # Save train data as CSV
    train_path = os.path.join(output_dir, '60_train_df_scaled_.csv')
    train_df_scaled.to_csv(train_path, index=False)
    print(f"✓ Saved training data: {train_path}")

    train_path_not_scaled = os.path.join(output_dir, '60__train_df_not_scaled_.csv')
    train_df.to_csv(train_path_not_scaled, index=False)
    print(f"✓ Saved training data: {train_path}")

    # Save validation data as CSV
    val_path = os.path.join(output_dir, '60_val_df_scaled.csv')
    val_df_scaled.to_csv(val_path, index=False)
    print(f"✓ Saved validation data: {val_path}")

    # Save validation data as CSV
    val_path = os.path.join(output_dir, '60_val_df_not_scaled.csv')
    val_df.to_csv(val_path, index=False)
    print(f"✓ Saved validation data: {val_path}")


    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler_60_patients.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler: {scaler_path}")

    # Verification - load and check saved files
    # Verification - load and check saved files
    print("\n6. VERIFICATION")
    print("-" * 40)

    loaded_train = pd.read_csv(train_path)
    loaded_val = pd.read_csv(val_path)

    with open(scaler_path, 'rb') as f:
        loaded_scaler = pickle.load(f)

    print("✓ All files loaded successfully")
    print(f"  Train shape: {loaded_train.shape}")
    print(f"  Val shape: {loaded_val.shape}")
    print(f"  Scaler type: {type(loaded_scaler).__name__}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_samples = len(loaded_train) + len(loaded_val)
    train_pct = len(loaded_train) / total_samples * 100
    val_pct = len(loaded_val) / total_samples * 100

    print(f"✓ Data preprocessing complete")
    print(f"✓ Single train/val split generated")
    print(f"✓ Split ratio: {train_pct:.0f}% train / {val_pct:.0f}% validation")
    print(f"✓ Total samples: {total_samples}")
    print(f"  - Training: {len(loaded_train)} samples ({loaded_train['user_id'].nunique()} users)")
    print(f"  - Validation: {len(loaded_val)} samples ({loaded_val['user_id'].nunique()} users)")
    print(f"✓ Features scaled with MinMaxScaler")
    print(f"✓ All files saved in: {output_dir}/")
    print("\nFiles created:")
    print(f"  - {output_dir}/train_df.pkl")
    print(f"  - {output_dir}/val_df.pkl")
    print(f"  - {output_dir}/scaler.pkl")


if __name__ == "__main__":
    main()