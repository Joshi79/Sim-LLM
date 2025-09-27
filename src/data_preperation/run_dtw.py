import pandas as pd
import os
from dtw_similarity import dtw_similarity_report, summarize_distances, mean_pairwise_between_test_and_train, build_patient_series

output_dir = "dtw_results"
os.makedirs(output_dir, exist_ok=True)

train = pd.read_csv("../../data/original_training_dataset.csv")
test = pd.read_csv("../../data/test_df.csv")

print(f"Train: {len(train)} rows, {train['user_id'].nunique()} patients")
print(f"Test: {len(test)} rows, {test['user_id'].nunique()} patienst")

leakage_df, nn_df, train_nn_df, baseline_df = dtw_similarity_report(
    train_df=train,
    test_df=test,
    id_col="user_id",
    time_col="day_part_x",
    feature_cols=['numberRating','highestRating', 'lowestRating', 'medianRating', 'sdRating',
                  'numberLowRating', 'numberMediumRating', 'numberHighRating',
                  'numberMessageReceived', 'numberMessageRead','action', 'reward'],
    normalize=True,
    max_within_pairs=200,
    zscore_per_patient=True
)


# Add these lines to call the new function
print("\nCalculating all-pairs DTW between test and train...")
all_pairs_df = mean_pairwise_between_test_and_train(
    train_series=build_patient_series(train, "user_id", "day_part_x", ['numberRating','highestRating', 'lowestRating', 'medianRating', 'sdRating', 'numberMessageReceived', 'numberMessageRead','action', 'reward']),
    test_series=build_patient_series(test, "user_id", "day_part_x", ['numberRating','highestRating', 'lowestRating', 'medianRating', 'sdRating', 'numberMessageReceived', 'numberMessageRead','action', 'reward'])
)
print(all_pairs_df)




print("\n" + "="*60)
print("DTW SIMILARITY RESULTS")
print("="*60)

print("\n LEAKAGE CHECK:")
print(leakage_df.to_string(index=False))

print(f"\n NEAREST DTW Train->TRAIN ({len(nn_df)} rows):")
print(train_nn_df)

# Now you can compare:
test_to_train_mean = nn_df['dtw_min'].mean()
train_to_train_mean = train_nn_df['dtw_min'].mean()

print(f"  Test->Train mean DTW:  {test_to_train_mean:.4f}")
print(f"  Train->Train mean DTW: {train_to_train_mean:.4f}")

print(f"\n NEAREST DTW SUMMARY:")
nn_summary = summarize_distances(nn_df['dtw_min'])
for key, value in nn_summary.items():
    print(f"  {key:>6}: {value}")

if not baseline_df.empty:
    print(f"\n BASELINE SUMMARY:")
    baseline_summary = summarize_distances(baseline_df['dtw'])
    for key, value in baseline_summary.items():
        print(f"  {key:>6}: {value}")

print(f"\n save output")
leakage_df.to_csv(os.path.join(output_dir, "dtw_leakage_check.csv"), index=False)
nn_df.to_csv(os.path.join(output_dir, "dtw_test_to_train_nearest.csv"), index=False)
baseline_df.to_csv(os.path.join(output_dir, "dtw_within_train_baseline.csv"), index=False)

