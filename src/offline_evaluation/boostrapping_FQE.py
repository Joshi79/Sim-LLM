
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from copy import deepcopy
from joblib import Parallel, delayed

from .FQE_def import perform_FQE
from . import utils


from DDQN import DDQNAgent
from FQE_def import perform_FQE
import utils




ALL_POLICIES_CONFIG = {
    "original_30": {
        "policy_type": "learned",
        "agent_path": "../../reports/hypertuned_ddqn/original_data_tuned.pth",
        "description": "Original (30 patients)"
    },
    "synthetic_60": {
        "policy_type": "learned",
        "agent_path": "../../reports/hypertuned_ddqn/synthetic_60_tuned.pth",
        "description": "Synthetic (60 patients)"
    },
    "synthetic_100": {
        "policy_type": "learned",
        "agent_path": "../../reports/hypertuned_ddqn/synthetic_100_tuned.pth",
        "description": "Synthetic (100 patients)"
    },
    "merged_120": {
        "policy_type": "learned",
        "agent_path": "../../reports/hypertuned_ddqn/merged_120_tuned.pth",
        "description": "Merged (120 patients)"
    },

    # Fixed policies
    "fixed_action_0": {
        "policy_type": "fixed",
        "fixed_action": 0,
        "description": "Fixed: No message"
    },
    "fixed_action_1": {
        "policy_type": "fixed",
        "fixed_action": 1,
        "description": "Fixed: Encouraging"
    },
    "fixed_action_2": {
        "policy_type": "fixed",
        "fixed_action": 2,
        "description": "Fixed: Informing"
    },
    "fixed_action_3": {
        "policy_type": "fixed",
        "fixed_action": 3,
        "description": "Fixed: Affirming"
    }
}




# change the path to the your testing path !!!
TEST_DATA_PATH = "../../data/test_df.csv"

B = 150
CI_LEVEL = 0.95
SEED = 42
N_JOBS = 14
DEVICE = "cpu"
SAVE_ROOT = "/home/joshi79/Projects/master_thesis/DDQN_FQE/final/final_data/unified_bootstrap_results"
ACTION_LABELS = ['No message', 'Encouraging', 'Informing', 'Affirming']


# Boostrapping made for parallel execution
def _single_bootstrap_run_unified(idx,episodes,policies_config,seed,device):
    """
    Run one bootstrap replicate for ALL policies using the SAME resampled episodes.
    Returns dictionary with policy_id -> value
    """

    rng = np.random.default_rng(seed)
    K = len(episodes)

    # Resample all 7 episodes with replacement
    res_idx = rng.integers(0, K, size=K)
    boot_df = pd.concat([episodes[i] for i in res_idx], ignore_index=True)

    rb_boot = utils._build_rb(boot_df, utils.ReplayBuffer, device=device)

    results = {}

    # Evaluate each policy on the SAME bootstrap sample
    for policy_id, config in policies_config.items():
        try:
            # Load policy
            agent, _ = utils.load_policy(config, device)

            # Deep copy the policy
            policy_copy = deepcopy(agent)

            # Run FQE
            v_list, _ = perform_FQE(
                rb_boot,
                policy_copy,
                boot_df,
                bool_learned_policy=True
            )

            results[policy_id] = float(v_list[-1])
        except Exception as e:
            print(f"Error in bootstrap {idx} for policy {policy_id}: {e}")
            results[policy_id] = np.nan

    return results


def unified_bootstrap_fqe_parallel(df,policies_config,B= 150,ci_level= 0.95,seed= 42,n_jobs= None,device= "cpu"):
    """
    Perform bootstrap FQE for multiple policies using the SAME bootstrap samples.
    """

    if n_jobs is None or n_jobs < 0:
        n_jobs = os.cpu_count()

    episodes = utils._split_episodes(df)
    n_episodes = len(episodes)

    print(f"Dataset contains {n_episodes} episodes (patients)")

    # Initialize results storage
    results = {}
    for policy_id, config in policies_config.items():
        results[policy_id] = {
            "description": config["description"],
            "policy_type": config["policy_type"],
            "bootstrap_values": [],
            "point_estimate": None,
            "action_distribution": None
        }

    # 2) Compute point estimates on full data
    print("\nComputing point estimates on full dataset.")
    for policy_id, config in policies_config.items():
        print(f"  {config['description']}.")
        agent, checkpoint = utils.load_policy(config, device)

        point_estimate = utils.compute_point_estimate(df, agent, device)
        results[policy_id]["point_estimate"] = float(point_estimate)

        action_dist = utils.compute_policy_action_distribution(agent, df)
        results[policy_id]["action_distribution"] = action_dist

        results[policy_id].update(
            utils.summarize_checkpoint_for_json(checkpoint, config, config["policy_type"]))

    # 3) Run bootstrap with SAME resampling for all policies
    print(f"\nStarting unified bootstrap evaluation (B={B}, n_jobs={n_jobs}).")

    # Prepare seeds for each replicate
    seeds = np.arange(seed, seed + B)

    t0 = time.time()

    # Run parallel bootstrap
    bootstrap_results = Parallel(n_jobs=n_jobs, verbose=5, backend="loky")(
        delayed(_single_bootstrap_run_unified)(
            i,
            episodes,
            policies_config,
            int(s),
            device
        )
        for i, s in enumerate(seeds)
    )

    elapsed = time.time() - t0

    for policy_id in policies_config.keys():
        values = [res[policy_id] for res in bootstrap_results if not np.isnan(res.get(policy_id, np.nan))]
        results[policy_id]["bootstrap_values"] = [float(v) for v in values]  # JSON-safe list


    for policy_id in results:
        samples = np.asarray(results[policy_id]["bootstrap_values"], dtype=float)
        point_est = float(results[policy_id]["point_estimate"])

        if len(samples) == 0:
            continue

        # Calculate epsilon values (errors)
        epsilon_values = samples - point_est
        epsilon_bar = np.mean(epsilon_values)

        # Bootstrap variance
        bootstrap_variance = np.sum((epsilon_values - epsilon_bar) ** 2) / (len(samples) - 1)

        # Confidence intervals (error-percentile & value-percentile)
        alpha = 1 - ci_level
        q_lower = np.percentile(epsilon_values, (alpha / 2) * 100)
        q_upper = np.percentile(epsilon_values, (1 - alpha / 2) * 100)

        error_ci_lower = float(point_est - q_upper)
        error_ci_upper = float(point_est - q_lower)

        value_ci_lower = float(np.percentile(samples, (alpha / 2) * 100))
        value_ci_upper = float(np.percentile(samples, (1 - alpha / 2) * 100))

        # Additional CIs
        additional_cis = {
            "80%": [float(np.percentile(samples, 10)), float(np.percentile(samples, 90))],
            "95%": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
            "99%": [float(np.percentile(samples, 0.5)), float(np.percentile(samples, 99.5))]
        }

        # Store results (JSON-safe)
        results[policy_id].update({
            "epsilon_values": epsilon_values.tolist(),
            "epsilon_bar": float(epsilon_bar),
            "bootstrap_variance": float(bootstrap_variance),
            "bootstrap_std": float(np.sqrt(bootstrap_variance)),
            "bootstrap_mean": float(np.mean(samples)),
            "bootstrap_median": float(np.median(samples)),
            "confidence_intervals": {
                "error_percentile": {
                    "method": "error_percentile",
                    "description": "Recommended by FQE bootstrap literature",
                    "ci": [error_ci_lower, error_ci_upper],
                    "ci_level": ci_level
                },
                "value_percentile": {
                    "method": "value_percentile",
                    "description": "Classical bootstrap percentile method",
                    "ci": [value_ci_lower, value_ci_upper],
                    "ci_level": ci_level
                },
                "additional_value_percentile_cis": additional_cis
            },
            "ci_width": float(error_ci_upper - error_ci_lower),
            "extended_statistics": utils.extended_stats(samples),
            "all_bootstrap_values": samples.tolist()
        })

    print("\nCalculating pairwise correlations.")
    correlations = {}
    policy_ids = list(policies_config.keys())

    for i, policy1 in enumerate(policy_ids):
        for j, policy2 in enumerate(policy_ids):
            if i <= j:  # Include diagonal
                eps1 = np.asarray(results[policy1]["bootstrap_values"], dtype=float) - float(results[policy1]["point_estimate"])
                eps2 = np.asarray(results[policy2]["bootstrap_values"], dtype=float) - float(results[policy2]["point_estimate"])

                if len(eps1) == 0 or len(eps2) == 0:
                    continue

                eps_bar1 = np.mean(eps1)
                eps_bar2 = np.mean(eps2)

                # Calculate correlation
                numerator = np.sum((eps1 - eps_bar1) * (eps2 - eps_bar2))
                denom1 = np.sqrt(np.sum((eps1 - eps_bar1) ** 2))
                denom2 = np.sqrt(np.sum((eps2 - eps_bar2) ** 2))

                if denom1 > 0 and denom2 > 0:
                    correlation = numerator / (denom1 * denom2)
                else:
                    correlation = np.nan

                key = f"{policy1}_vs_{policy2}"
                correlations[key] = {
                    "policy1": results[policy1]["description"],
                    "policy2": results[policy2]["description"],
                    "correlation": float(correlation) if not np.isnan(correlation) else None
                }

    return {
        "policies": results,
        "correlations": correlations,
        "bootstrap_parameters": {
            "B": B,
            "ci_level": ci_level,
            "seed": seed,
            "n_episodes": n_episodes,
            "n_policies": len(policies_config)
        },
        "elapsed_seconds": elapsed,
        "test_data_path": os.path.abspath(TEST_DATA_PATH)
    }


def main():
    """Main function to run unified bootstrap FQE for all policies"""

    utils.set_global_seeds(SEED)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # Load test data
    print(f"Loading test data: {TEST_DATA_PATH}")
    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"Test data shape: {test_df.shape}")

    # Run unified bootstrap FQE
    print(f"\nStarting unified bootstrap FQE for {len(ALL_POLICIES_CONFIG)} policies")
    print(f"Configuration: B={B}, CI_LEVEL={CI_LEVEL}, SEED={SEED}, N_JOBS={N_JOBS}")

    results = unified_bootstrap_fqe_parallel(
        df=test_df,
        policies_config=ALL_POLICIES_CONFIG,
        B=B,
        ci_level=CI_LEVEL,
        seed=SEED,
        n_jobs=N_JOBS,
        device=DEVICE
    )

    results["generation_timestamp"] = datetime.now().isoformat()

    # Print results summary
    print("\n" + "=" * 80)
    print("UNIFIED BOOTSTRAP FQE RESULTS")
    print("=" * 80)

    for policy_id, policy_results in results["policies"].items():
        print(f"\n{policy_results['description']}:")
        print(f"  Policy Type: {policy_results['policy_type']}")
        print(f"  Point Estimate: {policy_results['point_estimate']:.6f}")
        print(f"  Bootstrap Mean: {policy_results.get('bootstrap_mean', 'N/A')}")
        print(f"  Bootstrap Variance: {policy_results.get('bootstrap_variance', 'N/A')}")
        if 'confidence_intervals' in policy_results:
            error_ci = policy_results['confidence_intervals']['error_percentile']['ci']
            print(f"  {int(CI_LEVEL * 100)}% CI (Error-Percentile): [{error_ci[0]:.6f}, {error_ci[1]:.6f}]")
        print(f"  Action Distribution: {policy_results['action_distribution']}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save main unified results (JSON-safe)
    json_path = os.path.join(SAVE_ROOT, f"unified_bootstrap_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMain results saved to: {json_path}")

    for policy_id, policy_results in results["policies"].items():
        policy_dir = os.path.join(SAVE_ROOT, f"{policy_id}_{timestamp}")
        os.makedirs(policy_dir, exist_ok=True)

        individual_json = {
            "data_type": "unified_bootstrap_evaluation",
            "policy_id": policy_id,
            **policy_results,
            "bootstrap_parameters": results["bootstrap_parameters"],
            "generation_timestamp": results["generation_timestamp"]
        }

        individual_json_path = os.path.join(policy_dir, f"bootstrap_fqe_enhanced_{policy_id}.json")
        with open(individual_json_path, "w") as f:
            json.dump(individual_json, f, indent=2)

        if len(policy_results["bootstrap_values"]) > 0:
            npy_path = os.path.join(policy_dir, f"bootstrap_values_{policy_id}.npy")
            np.save(npy_path, np.asarray(policy_results["bootstrap_values"], dtype=float))

    print(f"\nTotal execution time: {results['elapsed_seconds']:.2f} seconds")
    print("\nUnified bootstrap evaluation completed successfully!")


if __name__ == "__main__":
    main()
