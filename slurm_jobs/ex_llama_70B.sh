#!/bin/bash
#SBATCH --job-name=lama3_70b_corrected
#SBATCH --time=15:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=logs/llama_without_statistics%j.out
#SBATCH --error=logs/llama_without_orginal_statistics%j.err


source $HOME/.bashrc
conda activate master_thesis_env

cd $HOME/Sim-LLM

# Ensure logs/ exists for SLURM output
mkdir -p logs

python -m simulation_data.simulations.run_simulations_without_synthetic_information
