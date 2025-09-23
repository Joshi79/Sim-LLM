import pandas as pd
from typing import List
import numpy as np
import json
from scipy import stats, signal
import matplotlib.pyplot as plt


def calculate_full_correlation_matrices(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                        all_cols: List[str], patient_col: str = "user_id") -> dict:
    """
    Compute full correlation matrices for trend/cycle components across all variable pairs.
    Following Algorithm 2 from the paper.
    """

    def extract_trend_cycle(series: pd.Series):
        x = series.to_numpy(dtype=float)
        cycle = signal.detrend(x)  # remove linear trend
        trend = x - cycle  # recover linear trend component
        return pd.Series(trend, index=series.index), pd.Series(cycle, index=series.index)

    def calculate_correlation_matrix_for_dataset(df: pd.DataFrame) -> dict:
        n_vars = len(all_cols)

        # Initialize correlation matrices
        trend_corr_matrix = pd.DataFrame(np.zeros((n_vars, n_vars)),
                                         index=all_cols, columns=all_cols)
        cycle_corr_matrix = pd.DataFrame(np.zeros((n_vars, n_vars)),
                                         index=all_cols, columns=all_cols)

        np.fill_diagonal(trend_corr_matrix.values, 1.0)
        np.fill_diagonal(cycle_corr_matrix.values, 1.0)

        if patient_col not in df.columns:
            return {"trend_matrix": trend_corr_matrix, "cycle_matrix": cycle_corr_matrix}


        trend_patient_corrs = {(i, j): [] for i in all_cols for j in all_cols if i != j}
        cycle_patient_corrs = {(i, j): [] for i in all_cols for j in all_cols if i != j}


        for _, patient_data in df.groupby(patient_col):
            if len(patient_data) < 3:
                continue

            patient_data = patient_data.sort_index()

            # extract trends and cycles for all variables
            trends, cycles = {}, {}
            for col in all_cols:
                if col in patient_data.columns:
                    if patient_data[col].notna().sum() >= 3:
                        t, c = extract_trend_cycle(patient_data[col].astype(float))
                        trends[col] = t
                        cycles[col] = c

            # Calculate correlations between all variable pairs
            for col_i in all_cols:
                if col_i not in trends:
                    continue

                for col_j in all_cols:
                    if col_i == col_j or col_j not in trends:
                        continue

                    # Trend correlations
                    if trends[col_i].std() > 1e-6 and trends[col_j].std() > 1e-6:
                        try:
                            tau_trend = trends[col_i].corr(trends[col_j], method="kendall")
                            if pd.notna(tau_trend):
                                trend_patient_corrs[(col_i, col_j)].append(tau_trend)
                        except:
                            pass

                    # Cycle correlations
                    if cycles[col_i].std() > 1e-6 and cycles[col_j].std() > 1e-6:
                        try:
                            tau_cycle = cycles[col_i].corr(cycles[col_j], method="kendall")
                            if pd.notna(tau_cycle):
                                cycle_patient_corrs[(col_i, col_j)].append(tau_cycle)
                        except:
                            pass

        # Average across patients
        for col_i in all_cols:
            for col_j in all_cols:
                if col_i != col_j:
                    key = (col_i, col_j)
                    if key in trend_patient_corrs and trend_patient_corrs[key]:
                        trend_corr_matrix.loc[col_i, col_j] = np.mean(trend_patient_corrs[key])
                    if key in cycle_patient_corrs and cycle_patient_corrs[key]:
                        cycle_corr_matrix.loc[col_i, col_j] = np.mean(cycle_patient_corrs[key])

        return {"trend_matrix": trend_corr_matrix, "cycle_matrix": cycle_corr_matrix}

    real_results = calculate_correlation_matrix_for_dataset(real_df)
    syn_results = calculate_correlation_matrix_for_dataset(synthetic_df)

    return {
        "trend_real": real_results["trend_matrix"],
        "trend_synthetic": syn_results["trend_matrix"],
        "cycle_real": real_results["cycle_matrix"],
        "cycle_synthetic": syn_results["cycle_matrix"]
    }


def create_temporal_correlation_heatmap(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                        all_cols: List[str], patient_col: str = "user_id",
                                        pretty: dict | None = None):
    """
    Create correlation heatmaps for trend and cycle components comparison.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Calculate full correlation matrices
    corr_matrices = calculate_full_correlation_matrices(real_df, synthetic_df, all_cols, patient_col)
    n_vars = len(corr_matrices["trend_synthetic"])
    mask = np.triu(np.ones((n_vars, n_vars), dtype=bool), k=1)

    if pretty is not None:
        pretty_labels = [pretty.get(col, col) for col in all_cols]
        for key in corr_matrices:
            corr_matrices[key].index = pretty_labels
            corr_matrices[key].columns = pretty_labels

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Synthetic data trends
    sns.heatmap(corr_matrices["trend_synthetic"], ax=axes1[0], mask=mask,
                cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True,
                cbar_kws={'label': 'Kendall τ'})
    axes1[0].set_title('Synthetic Data', fontsize=14,weight='bold')
    axes1[0].set_xticklabels(axes1[0].get_xticklabels(), rotation=45, ha='right', fontsize=12,  fontweight='bold')
    axes1[0].set_yticklabels(axes1[0].get_yticklabels(), rotation=0, fontsize=12, fontweight='bold')

    # Right: Real data trends
    sns.heatmap(corr_matrices["trend_real"], ax=axes1[1], mask=mask,
                cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True,
                cbar_kws={'label': 'Kendall τ'})
    axes1[1].set_title('Test Data', fontsize=14, weight='bold')
    axes1[1].set_xticklabels(axes1[1].get_xticklabels(), rotation=45, ha='right', fontsize=12, fontweight='bold')
    axes1[1].set_yticklabels(axes1[1].get_yticklabels(), rotation=0, fontsize=12, fontweight='bold')
    plt.tight_layout()

    # Save trend figure
    plt.savefig('../output_data_fidelity/temporal_correlation_trends_test_basline_prompt.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    #plt.show()

    # Cycle correlations heatmap
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(corr_matrices["cycle_synthetic"], ax=axes2[0], mask=mask,
                cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True,
                cbar_kws={'label': 'Kendall τ'})
    axes2[0].set_title('Synthetic Data', fontsize=14, weight='bold')
    axes2[0].set_xticklabels(axes2[0].get_xticklabels(), rotation=45, ha='right', fontsize=12, fontweight='bold')
    axes2[0].set_yticklabels(axes2[0].get_yticklabels(), rotation=0, fontsize=12 ,fontweight='bold')

    sns.heatmap(corr_matrices["cycle_real"], ax=axes2[1], mask=mask,
                cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True,
                cbar_kws={'label': 'Kendall τ'})
    axes2[1].set_title('Test Data', fontsize=14, weight='bold')
    axes2[1].set_xticklabels(axes2[1].get_xticklabels(), rotation=45, ha='right', fontsize=12, fontweight='bold')
    axes2[1].set_yticklabels(axes2[1].get_yticklabels(), rotation=0, fontsize=12,fontweight='bold')

    plt.tight_layout()

    # Save cycle figure
    plt.savefig('../output_data_fidelity/temporal_correlation_cycles_test_baseline_prompt.pdf', format='pdf', dpi=1200, bbox_inches='tight')

    # Calculate and print correlation differences
    trend_diff = np.abs(corr_matrices["trend_real"].values - corr_matrices["trend_synthetic"].values)
    cycle_diff = np.abs(corr_matrices["cycle_real"].values - corr_matrices["cycle_synthetic"].values)

    # Exclude diagonal (self-correlations) from difference calculations
    mask = ~np.eye(trend_diff.shape[0], dtype=bool)
    avg_trend_diff = np.nanmean(trend_diff[mask])
    max_trend_diff = np.nanmax(trend_diff[mask])
    avg_cycle_diff = np.nanmean(cycle_diff[mask])
    max_cycle_diff = np.nanmax(cycle_diff[mask])

    print(f"\nTemporal Correlation Matrix Comparison:")
    print(f"  Trend Components:")
    print(f"    Average absolute difference: {avg_trend_diff:.4f}")
    print(f"    Maximum absolute difference: {max_trend_diff:.4f}")
    print(f"  Cycle Components:")
    print(f"    Average absolute difference: {avg_cycle_diff:.4f}")
    print(f"    Maximum absolute difference: {max_cycle_diff:.4f}")

    return corr_matrices

# original data paths
original_data_splits = r"C:\Users\User\PycharmProjects\master_thesis\simulation_data\final_run_data_preparation\data_splits\original_training_dataset.csv"
real_test_path = r"C:\Users\User\PycharmProjects\master_thesis\simulation_data\final_run_data_preparation\data_splits\test_df.csv"


# synthetic data paths
synthetic_path_no_orignal_data_info = r"C:\Users\User\PycharmProjects\Sim-LLM\data\prompt_no_info_orignal_data.csv"
synthetic_path_not_grounded_in_synthetic = r"C:\Users\User\PycharmProjects\Sim-LLM\data\prompt_not_grounded_in_synthetic.csv"
synthetic_baseline = r"C:\Users\User\PycharmProjects\Sim-LLM\data\synthetic_data_baseline_prompt.csv"


pretty = {
        'numberRating': 'No. of ratings',
        'highestRating': 'Highest rating',
        'lowestRating': 'Lowest rating',
        'medianRating': 'Median rating',
        'sdRating': 'SD rating',
        'numberLowRating': 'No. low ratings',
        'numberMediumRating': 'No. medium Ratings',
        'numberHighRating': 'No. high ratings',
        'numberMessageRead': 'No. messages read',
        'readAllMessage': 'Read all messages',
        'numberMessageReceived' : 'No. messages received'
    }

categorical_cols = [
            'numberRating', 'highestRating', 'lowestRating',
            'numberLowRating', 'numberMediumRating', 'numberHighRating',
            'numberMessageRead', 'readAllMessage', 'numberMessageReceived', "medianRating"
        ]

continuous_cols = ['sdRating']
all_columns = categorical_cols + continuous_cols

synthetic_df_no_info = pd.read_csv(synthetic_path_no_orignal_data_info)
synthetic_df_no_grounding = pd.read_csv(synthetic_path_not_grounded_in_synthetic)
synthetic_df = pd.read_csv(synthetic_baseline)
real_df_original = pd.read_csv(original_data_splits)
real_df_test = pd.read_csv(real_test_path)

create_temporal_correlation_heatmap(
    real_df=real_df_test,
    synthetic_df=synthetic_df,
    all_cols=all_columns,
    patient_col="user_id",
    pretty=pretty
)

trends = calculate_full_correlation_matrices(
    real_df=real_df_test,
    synthetic_df=synthetic_df,
    all_cols=all_columns,
    patient_col="user_id"
)

trends_json = {
    "trend_real": trends["trend_real"].to_dict(),
    "trend_synthetic": trends["trend_synthetic"].to_dict(),
    "cycle_real": trends["cycle_real"].to_dict(),
    "cycle_synthetic": trends["cycle_synthetic"].to_dict()
}

with open("../../reports/output_data_fidelity/temporal_correlation_synthetic_baseline.json", "w") as f:
    json.dump(trends_json, f, indent=4)

