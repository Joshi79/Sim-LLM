#!/bin/bash
#SBATCH --job-name=optuna_DDQN
#SBATCH --time=72:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/optuna_DDQN%j.out
#SBATCH --error=logs/Optunda_DDQN%j.err

source $HOME/.bashrc
conda activate master_thesis_env

cd $HOME/Sim-LLM

# Ensure logs/ exists for SLURM output
mkdir -p logs

python -m src.offline_evaluation.optuna_tuning_DDQN
