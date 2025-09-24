import pandas as pd
import os
from dtw_similarity import dtw_similarity_report, summarize_distances

output_dir = "dtw_results"
os.makedirs(output_dir, exist_ok=True)

train = pd.read_csv("../../data/train_df.csv")
test = pd.read_csv("../../data/test_df.csv")

print(f"Train: {len(train)} rows, {train['user_id'].nunique()} patients")
print(f"Test: {len(test)} rows, {test['user_id'].nunique()} patienst")

leakage_df, nn_df, baseline_df = dtw_similarity_report(
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

print("\n" + "="*60)
print("DTW SIMILARITY RESULTS")
print("="*60)

print("\n LEAKAGE CHECK:")
print(leakage_df.to_string(index=False))


print(f"\n NEAREST DTW SUMMARY:")
nn_summary = summarize_distances(nn_df['dtw_min'])
for key, value in nn_summary.items():
    print(f"  {key:>6}: {value}")

if not baseline_df.empty:
    print(f"\n BASELINE SUMMARY:")
    baseline_summary = summarize_distances(baseline_df['dtw'])
    for key, value in baseline_summary.items():
        print(f"  {key:>6}: {value}")

# SPEICHERE ERGEBNISSE
print(f"\n save output")
leakage_df.to_csv(os.path.join(output_dir, "dtw_leakage_check.csv"), index=False)
nn_df.to_csv(os.path.join(output_dir, "dtw_test_to_train_nearest.csv"), index=False)
baseline_df.to_csv(os.path.join(output_dir, "dtw_within_train_baseline.csv"), index=False)

