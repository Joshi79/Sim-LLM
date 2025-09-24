#!/bin/bash
#SBATCH --job-name=lama3_8B_Instruct
#SBATCH --time=00:05:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=logs/lama3_1_8B_Instruct_cached_with_data_checks_out_h100_%j.out
#SBATCH --error=logs/lama3_8B_cached_with_data_checks_out_h100_%j.err

source $HOME/.bashrc
conda activate master_thesis_env

cd $HOME/Sim-LLM

# Ensure logs/ exists for SLURM output
mkdir -p logs

python -m src.simulation.run_simulations.run_simulation_8B
