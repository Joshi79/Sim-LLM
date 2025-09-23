import os
import sys
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import optuna
import joblib
from collections import deque
from datetime import datetime
from pathlib import Path
import logging

from src.offline_evaluation.DDQN import DDQNAgent
from src.offline_evaluation.FQE_def import perform_FQE
import src.offline_evaluation.utils as utils
from src.offline_evaluation.utils import COLUMNS_RL_ALGORITHM

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/optuna_tuning.log')
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Constants
STATE_DIM = 12
ACTION_DIM = 4
BUFFER_SIZE = 1e5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


#

DATASETS = {
    "dataset_60_patients": {
        "train": "data/data_splits_syn/60_train_df_scaled_.csv",
        "val": "data/data_splits_syn/60_val_df_scaled.csv"
    },
    "dataset_100_patients": {
        "train": "data/data_splits_syn/full_train_df_scaled_.csv",
        "val": "data/data_splits_syn/full_val_df_not_scaled.csv"
    },
    "dataset_merged": {
        "train": "data/data_splits_syn/merged_train_original_val_scaled.csv",
        "val": "data/data_splits_syn/merged_val_original_val_scaled.csv"
    },
    "dataset_original": {
        "train": "data/data_splits/train_df_scaled.csv",
        "val": "data/data_splits/val_df_scaled.csv"
    }
}





# create output directory to store the tuning results
OUTPUT_DIR = Path("output/tuning_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_dataset(train_path, val_path):
    """Load and prepare a dataset for training."""
    logger.info(f"  Loading train data from: {train_path}")
    train_df = pd.read_csv(train_path)
    logger.info(f"    Train samples: {len(train_df)}")

    logger.info(f"  Loading val data from: {val_path}")
    val_df = pd.read_csv(val_path)
    logger.info(f"    Val samples: {len(val_df)}")

    # convert the data in the correct format for the RL algorithm
    train_dict = utils.get_format_data_rl_algorithm(train_df)
    val_dict = utils.get_format_data_rl_algorithm(val_df)

    return train_df, val_df, train_dict, val_dict


def create_replay_buffers(train_dict, val_dict, batch_size):
    """Create replay buffers for training and validation."""
    replay_buffer_train = utils.ReplayBuffer(
        STATE_DIM, batch_size, BUFFER_SIZE, DEVICE
    )
    replay_buffer_train.load_d4rl_dataset(train_dict)

    # Validation buffer
    replay_buffer_val = utils.ReplayBuffer(
        STATE_DIM, batch_size, BUFFER_SIZE, DEVICE
    )
    replay_buffer_val.load_d4rl_dataset(val_dict)

    return replay_buffer_train, replay_buffer_val


def objective_function(trial, train_dict, val_dict, val_df, dataset_name):
    """ Objective function for Optuna optimization."""

    # Hyperparameter suggestions
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])


    logger.info(f"\n  Trial {trial.number} for {dataset_name}")
    logger.info(f"    lr={learning_rate:.2e}, hidden={hidden_size}, batch={batch_size}")

    try:
        # Create replay buffers
        replay_buffer_train, replay_buffer_val = create_replay_buffers(
            train_dict, val_dict, batch_size
        )

        # Create DDQN agent
        agent = DDQNAgent(
            state_size=STATE_DIM,
            action_size=ACTION_DIM,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            device=DEVICE
        )

        # Update tau and gamma
        agent.tau = 1e-5
        agent.gamma = 0.9

        # Training parameters
        max_iterations = 30001
        eval_interval = 6000
        early_stopping_patience = 3
        min_delta = 0.001

        # Training loop
        best_val_fqe = -float('inf')
        no_improvement_count = 0
        loss_history = deque(maxlen=1000)

        for iteration in range(max_iterations):
            # Training step
            loss = agent.train(replay_buffer_train)
            loss_history.append(loss)

            # Evaluation
            if iteration % eval_interval == 0 and iteration > 0:
                # Compute FQE
                FQE_values, _ = perform_FQE(
                    replay_buffer=replay_buffer_val,
                    network_learned_policy=agent,
                    df=val_df,
                    bool_learned_policy=True
                )
                current_fqe = FQE_values[-1]

                avg_loss = np.mean(loss_history)
                logger.info(f"      Iter {iteration}: FQE={current_fqe:.4f}, Loss={avg_loss:.6f}")

                # Check for improvement
                if current_fqe > best_val_fqe + min_delta:
                    best_val_fqe = current_fqe
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Early stopping
                if no_improvement_count >= early_stopping_patience:
                    logger.info(f"      Early stopping at iteration {iteration}")
                    break

                # Report to Optuna for pruning
                trial.report(current_fqe, iteration)
                if trial.should_prune():
                    logger.info(f"      Trial pruned at iteration {iteration}")
                    raise optuna.TrialPruned()

        logger.info(f"      Final FQE: {best_val_fqe:.4f}")
        return best_val_fqe

    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"      Trial failed: {e}")
        return -1000.0


def optimize_dataset(dataset_name, train_path, val_path, n_trials=50):
    """Run hyperparameter optimization for a single dataset."""

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Optimizing Dataset: {dataset_name}")
    logger.info(f"{'=' * 60}")

    # Load dataset
    try:
        train_df, val_df, train_dict, val_dict = load_and_prepare_dataset(
            train_path, val_path
        )
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        return None

    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=30,
            n_warmup_steps=6000,
            interval_steps=6000
        ),
        sampler=optuna.samplers.TPESampler(n_startup_trials=20,
                                           seed=42)
    )

    # Run optimization
    logger.info(f"\nStarting optimization with {n_trials} trials...")

    def objective(trial):
        return objective_function(trial, train_dict, val_dict, val_df, dataset_name)

    study.optimize(objective, n_trials=n_trials)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save study
    study_path = OUTPUT_DIR / f"{dataset_name}_study_{timestamp}.pkl"
    joblib.dump(study, study_path)
    logger.info(f"\nStudy saved to: {study_path}")

    # Save best hyperparameters
    best_params = study.best_params
    best_params['best_fqe_value'] = study.best_value
    best_params['dataset'] = dataset_name
    best_params['timestamp'] = timestamp
    best_params['n_trials'] = len(study.trials)
    best_params['n_completed'] = len([t for t in study.trials
                                      if t.state == optuna.trial.TrialState.COMPLETE])
    best_params['n_pruned'] = len([t for t in study.trials
                                   if t.state == optuna.trial.TrialState.PRUNED])

    params_path = OUTPUT_DIR / f"{dataset_name}_best_params_{timestamp}.json"
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Best parameters saved to: {params_path}")

    # Print summary
    logger.info(f"\n{'-' * 40}")
    logger.info(f"Summary for {dataset_name}:")
    logger.info(f"  Best FQE: {study.best_value:.4f}")
    logger.info(f"  Trials: {best_params['n_completed']} completed, {best_params['n_pruned']} pruned")
    logger.info(f"  Best hyperparameters:")
    for param, value in study.best_params.items():
        if isinstance(value, float) and value < 0.01:
            logger.info(f"    {param}: {value:.2e}")
        elif isinstance(value, float):
            logger.info(f"    {param}: {value:.4f}")
        else:
            logger.info(f"    {param}: {value}")

    return study


def train_and_save_best_agent(best_params, train_dict, val_dict, val_df, dataset_name):

    logger.info(f"\nTraining final agent with best hyperparameters for {dataset_name}")

    # Create replay buffers with best batch size
    replay_buffer_train, replay_buffer_val = create_replay_buffers(
        train_dict, val_dict, best_params['batch_size']
    )

    # Create agent with best hyperparameters
    agent = DDQNAgent(
        state_size=STATE_DIM,
        action_size=ACTION_DIM,
        learning_rate=best_params['learning_rate'],
        hidden_size=best_params['hidden_size'],
        device=DEVICE
    )

    agent.tau = 1e-5
    agent.gamma = 0.9

    # Training parameters
    max_iterations = 30001
    eval_interval = 6000
    best_val_fqe = -float('inf')
    best_agent = None

    logger.info(f"  Training for {max_iterations} iterations...")

    for iteration in range(max_iterations):
        # Training step
        loss = agent.train(replay_buffer_train)

        # Evaluation and model saving
        if iteration % eval_interval == 0 and iteration > 0:
            FQE_values, _ = perform_FQE(
                replay_buffer=replay_buffer_val,
                network_learned_policy=agent,
                df=val_df,
                bool_learned_policy=True
            )
            current_fqe = FQE_values[-1]

            logger.info(f"    Iter {iteration}: FQE={current_fqe:.4f}")

            # Save best agent
            if current_fqe > best_val_fqe:
                best_val_fqe = current_fqe
                best_agent = copy.deepcopy(agent)

    # Save the best agent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    agent_path = OUTPUT_DIR / f"{dataset_name}_best_agent_{timestamp}.pth"

    torch.save({
        'model_state_dict': best_agent.network.state_dict(),
        'target_net_state_dict': best_agent.target_net.state_dict(),
        'hyperparameters': best_params,
        'final_fqe': best_val_fqe,
        'dataset': dataset_name
    }, agent_path)

    logger.info(f"  Best agent saved to: {agent_path}")
    logger.info(f"  Final FQE: {best_val_fqe:.4f}")

    return best_agent, agent_path


def main():
    logger.info("=" * 60)
    logger.info("MULTI-DATASET DDQN HYPERPARAMETER TUNING")
    logger.info("=" * 60)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Number of datasets: {len(DATASETS)}")

    n_trials = 50
    logger.info(f"Trials per dataset: {n_trials}")

    # Store all results
    all_results = {}
    trained_agents = {}

    # Run optimization for each dataset
    for dataset_name, paths in DATASETS.items():
        study = optimize_dataset(
            dataset_name=dataset_name,
            train_path=paths["train"],
            val_path=paths["val"],
            n_trials=n_trials
        )

        if study is not None:
            all_results[dataset_name] = {
                'best_value': study.best_value,
                'best_params': study.best_params,
                'n_trials': len(study.trials)
            }

            # Train and save best agent
            try:
                logger.info(f"\n{'='*40}")
                logger.info(f"TRAINING FINAL AGENT: {dataset_name}")
                logger.info(f"{'='*40}")

                # Load dataset again for training
                train_df, val_df, train_dict, val_dict = load_and_prepare_dataset(
                    paths["train"], paths["val"]
                )

                # Train and save best agent
                best_agent, agent_path = train_and_save_best_agent(
                    study.best_params, train_dict, val_dict, val_df, dataset_name
                )

                trained_agents[dataset_name] = {
                    'agent': best_agent,
                    'path': str(agent_path),
                    'hyperparameters': study.best_params
                }

            except Exception as e:
                logger.error(f"Failed to train final agent for {dataset_name}: {e}")

    if all_results:

        for dataset_name in all_results:
            if dataset_name in trained_agents:
                all_results[dataset_name]['agent_path'] = trained_agents[dataset_name]['path']

        summary_path = OUTPUT_DIR / f"all_datasets_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nCombined summary saved to: {summary_path}")

        # Print final comparison
        logger.info(f"\n{'=' * 60}")
        logger.info("FINAL COMPARISON")
        logger.info(f"{'=' * 60}")
        for dataset_name, results in all_results.items():
            logger.info(f"\n{dataset_name}:")
            logger.info(f"  Best FQE: {results['best_value']:.4f}")
            logger.info(f"  Best learning_rate: {results['best_params'].get('learning_rate', 'N/A'):.2e}")
            logger.info(f"  Best hidden_size: {results['best_params'].get('hidden_size', 'N/A')}")
            logger.info(f"  Best batch_size: {results['best_params'].get('batch_size', 'N/A')}")

    logger.info(f"\n{'='*60}")
    logger.info("TRAINED AGENTS SUMMARY")
    logger.info(f"{'='*60}")
    for dataset_name, agent_info in trained_agents.items():
        logger.info(f"\n{dataset_name}:")
        logger.info(f"  Agent saved to: {agent_info['path']}")
        logger.info(f"  Best hyperparameters used for training")

    logger.info(f"\n{'=' * 60}")
    logger.info("TUNING AND TRAINING COMPLETED")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()