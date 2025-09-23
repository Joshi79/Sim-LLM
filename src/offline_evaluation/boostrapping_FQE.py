
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

from DDQN import DDQNAgent
from FQE_def import perform_FQE
import utils


ALL_POLICIES_CONFIG = {
    "original_30": {
        "policy_type": "learned",
        "agent_path": "/home/joshi79/Projects/master_thesis/DDQN_FQE/final/final_data/final_agent_trained/dataset_original_best_agent_20250821_205253.pth",
        "description": "Original (30 patients)"
    },
    "synthetic_60": {
        "policy_type": "learned",
        "agent_path": "/home/joshi79/Projects/master_thesis/DDQN_FQE/final/final_data/final_agent_trained/dataset_60_patients_best_agent_20250821_044248.pth",
        "description": "Synthetic (60 patients)"
    },
    "synthetic_100": {
        "policy_type": "learned",
        "agent_path": "/home/joshi79/Projects/master_thesis/DDQN_FQE/final/final_data/final_agent_trained/dataset_100_patients_best_agent_20250821_095237.pth",
        "description": "Synthetic (100 patients)"
    },
    "merged_120": {
        "policy_type": "learned",
        "agent_path": "/home/joshi79/Projects/master_thesis/DDQN_FQE/final/final_data/final_agent_trained/dataset_merged_best_agent_20250821_154121.pth",
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




# change the path to the required test data path
TEST_DATA_PATH = "DDQN_FQE/final/test_df_scaled.csv"

# ---------------- Parameters ----------------
B = 150
CI_LEVEL = 0.95
SEED = 42
N_JOBS = 14
DEVICE = "cpu"
SAVE_ROOT = "/home/joshi79/Projects/master_thesis/DDQN_FQE/final/final_data/unified_bootstrap_results"
ACTION_LABELS = ['No message', 'Encouraging', 'Informing', 'Affirming']


# ---------------- Fixed Policy Agent ----------------
class FixedAgent:
    """Fixed policy that always selects the same action"""

    def __init__(self, fixed_action=0, device="cpu"):
        self.fixed_action = int(fixed_action)
        self.device = device
        self.action_size = 4
        self.network = None  # compatibility

    def select_action_batch_not_encoded(self, states):
        batch_size = int(states.shape[0])
        return torch.full((batch_size,), self.fixed_action, dtype=torch.long, device=self.device)

    def select_action_batch(self, states):
        idx = self.select_action_batch_not_encoded(states)
        return F.one_hot(idx, num_classes=self.action_size).to(dtype=torch.float32)

    def select_action(self, state):
        return self.fixed_action


# ---------------- Utilities ----------------
def _split_episodes(df: pd.DataFrame, id_col: str = "user_id") -> List[pd.DataFrame]:
    """Return a list of dataframes, one per episode (grouped by user_id)."""
    return [g.reset_index(drop=True) for _, g in df.groupby(id_col, observed=False)]


def _build_rb(df_ep: pd.DataFrame,
              rb_cls,
              state_dim: int = 12,
              batch_size: int = 64,
              device: str = "cpu"):
    """Build replay buffer from dataframe"""
    d = utils.get_format_data_rl_algorithm(df_ep)
    rb = rb_cls(state_dim=state_dim,
                batch_size=batch_size,
                buffer_size=len(df_ep) + 1,
                device=device)
    rb.load_d4rl_dataset(d)
    return rb


# ---------------- Loading Functions ----------------
def set_global_seeds(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_trained_agent(agent_path: str, device: str = "cpu") -> Tuple[DDQNAgent, Dict]:
    """Load trained DDQN agent from checkpoint"""
    print(f"  Loading DDQN agent from: {agent_path}")

    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"Agent file not found: {agent_path}")

    map_location = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    checkpoint = torch.load(agent_path, map_location=map_location, weights_only=False)

    hyperparams = checkpoint['hyperparameters']

    agent = DDQNAgent(
        state_size=12,
        action_size=4,
        learning_rate=hyperparams.get('learning_rate', 0.001),
        hidden_size=hyperparams.get('hidden_size', 128),
        device=device
    )

    agent.tau = hyperparams.get('tau', 0.005)
    agent.gamma = hyperparams.get('gamma', 0.99)

    agent.network.load_state_dict(checkpoint['model_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])

    if device == "cuda" and torch.cuda.is_available():
        agent.network = agent.network.to(device)
        agent.target_net = agent.target_net.to(device)

    return agent, checkpoint


def load_policy(config: Dict[str, Any], device: str = "cpu") -> Tuple[Union[DDQNAgent, FixedAgent], Dict]:
    """Load either a learned DDQN agent or create a fixed policy agent"""
    if config["policy_type"] == "learned":
        return load_trained_agent(config["agent_path"], device)
    elif config["policy_type"] == "fixed":
        agent = FixedAgent(fixed_action=config["fixed_action"], device=device)
        checkpoint = {
            "hyperparameters": {
                "policy_type": "fixed",
                "fixed_action": config["fixed_action"]
            }
        }
        return agent, checkpoint
    else:
        raise ValueError(f"Unknown policy type: {config['policy_type']}")


# ---- JSON-safe checkpoint summarization (mirrors enhanced_bootstrap_agent.py idea)
def summarize_checkpoint_for_json(checkpoint: Dict[str, Any],
                                  config: Dict[str, Any],
                                  policy_type: str) -> Dict[str, Any]:
    if policy_type == "learned":
        h = checkpoint.get("hyperparameters", {})
        return {
            "model_info": {
                "architecture": "DDQN",
                "state_size": 12,
                "action_size": 4,
                "hidden_size": int(h.get("hidden_size", 128))
            },
            "training_info": {
                "final_fqe": checkpoint.get("final_fqe", "N/A"),
                "dataset": checkpoint.get("dataset", "N/A"),
                "training_epochs": checkpoint.get("training_epochs", "N/A")
            },
            "hyperparameters": {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in h.items()},
            "agent_path": os.path.abspath(config.get("agent_path", "")),
        }
    else:
        # Fixed policy metadata
        h = checkpoint.get("hyperparameters", {})
        return {
            "model_info": {
                "architecture": "fixed_policy",
                "state_size": 12,
                "action_size": 4,
                "hidden_size": None
            },
            "training_info": {
                "final_fqe": "N/A",
                "dataset": "N/A",
                "training_epochs": "N/A"
            },
            "hyperparameters": h,
            "agent_path": None
        }


# ---------------- Evaluation helpers ----------------
def compute_point_estimate(df: pd.DataFrame,
                           agent: Union[DDQNAgent, FixedAgent],
                           device: str = "cpu") -> float:
    """Compute FQE point estimate on the full dataset"""
    d = utils.get_format_data_rl_algorithm(df)
    rb_full = utils.ReplayBuffer(
        state_dim=12,
        batch_size=64,
        buffer_size=len(df) + 1,
        device=device
    )
    rb_full.load_d4rl_dataset(d)

    agent_copy = deepcopy(agent)
    v_list, _ = perform_FQE(
        rb_full,
        agent_copy,
        df,
        bool_learned_policy=True
    )

    return float(v_list[-1])


def compute_policy_action_distribution(agent: Union[DDQNAgent, FixedAgent],
                                       df: pd.DataFrame) -> Dict[str, int]:
    """Compute action distribution for a policy"""
    formatted = utils.get_format_data_rl_algorithm(df)
    states = torch.tensor(formatted["states"], dtype=torch.float32,
                          device=getattr(agent, 'device', 'cpu'))
    with torch.no_grad():
        pred_idx = agent.select_action_batch_not_encoded(states)
    pred_idx_cpu = pred_idx.cpu().numpy()

    counts = {}
    for i in range(len(ACTION_LABELS)):
        counts[ACTION_LABELS[i]] = int((pred_idx_cpu == i).sum())
    return counts


def extended_stats(samples: np.ndarray) -> Dict[str, Any]:
    """Compute extended statistics for bootstrap samples"""
    s = samples
    return {
        "n_samples": int(len(s)),
        "mean": float(np.mean(s)),
        "std": float(np.std(s)),
        "median": float(np.median(s)),
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "range": float(np.max(s) - np.min(s)),
        "q25": float(np.percentile(s, 25)),
        "q75": float(np.percentile(s, 75)),
        "iqr": float(np.percentile(s, 75) - np.percentile(s, 25)),
        "skewness": float(pd.Series(s).skew()),
        "kurtosis": float(pd.Series(s).kurtosis()),
        "se": float(np.std(s) / np.sqrt(len(s))),
        "cv": float(np.std(s) / abs(np.mean(s))) if np.mean(s) != 0 else float('inf')
    }


# ---------------- Modified Bootstrap Function ----------------
def _single_bootstrap_run_unified(idx: int,
                                  episodes: List[pd.DataFrame],
                                  policies_config: Dict[str, Dict],
                                  seed: int,
                                  device: str) -> Dict[str, float]:
    """
    Run one bootstrap replicate for ALL policies using the SAME resampled episodes.
    Returns dictionary with policy_id -> value
    """

    rng = np.random.default_rng(seed)
    K = len(episodes)

    # 1) Resample episodes with replacement (SAME for all policies)
    res_idx = rng.integers(0, K, size=K)
    boot_df = pd.concat([episodes[i] for i in res_idx], ignore_index=True)

    # 2) Build replay buffer once for the resample
    rb_boot = _build_rb(boot_df, utils.ReplayBuffer, device=device)

    results = {}

    # 3) Evaluate each policy on the SAME bootstrap sample
    for policy_id, config in policies_config.items():
        try:
            # Load policy
            agent, _ = load_policy(config, device)

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


def unified_bootstrap_fqe_parallel(
        df: pd.DataFrame,
        policies_config: Dict[str, Dict[str, Any]],
        B: int = 100,
        ci_level: float = 0.90,
        seed: int = 42,
        n_jobs: Optional[int] = None,
        device: str = "cpu"
) -> Dict[str, Any]:
    """
    Perform bootstrap FQE for multiple policies using the SAME bootstrap samples.
    """

    if n_jobs is None or n_jobs < 0:
        n_jobs = os.cpu_count()

    # 1) Split the dataset once into episodes
    episodes = _split_episodes(df)
    n_episodes = len(episodes)
    print(f"Dataset contains {n_episodes} episodes (patients)")

    # Initialize results storage
    results: Dict[str, Dict[str, Any]] = {}
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
        agent, checkpoint = load_policy(config, device)

        point_estimate = compute_point_estimate(df, agent, device)
        results[policy_id]["point_estimate"] = float(point_estimate)

        action_dist = compute_policy_action_distribution(agent, df)
        results[policy_id]["action_distribution"] = action_dist

        # JSON-safe metadata instead of raw checkpoint tensors
        results[policy_id].update(
            summarize_checkpoint_for_json(checkpoint, config, config["policy_type"])
        )

        print(f"    Point estimate: {point_estimate:.6f}")

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

    # 4) Organize results by policy (STORE AS LISTS)
    for policy_id in policies_config.keys():
        values = [res[policy_id] for res in bootstrap_results if not np.isnan(res.get(policy_id, np.nan))]
        results[policy_id]["bootstrap_values"] = [float(v) for v in values]  # JSON-safe list

    # 5) Calculate statistics and confidence intervals
    print("\nCalculating statistics and confidence intervals.")

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
            "extended_statistics": extended_stats(samples),
            "all_bootstrap_values": samples.tolist()
        })

    # 6) Calculate pairwise correlations (using SAME bootstrap samples)
    print("\nCalculating pairwise correlations.")
    correlations: Dict[str, Dict[str, Any]] = {}
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

    set_global_seeds(SEED)
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

    # Save individual policy results (for compatibility)
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
