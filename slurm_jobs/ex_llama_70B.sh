#!/bin/bash
#SBATCH --job-name=lama3_70b
#SBATCH --time=15:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=logs/llama_70B%j.out
#SBATCH --error=logs/llama_70B%j.err


source $HOME/.bashrc
conda activate master_thesis_env

cd $HOME/Sim-LLM

# Ensure logs/ exists for SLURM output
mkdir -p logs

python -m src.simulation.run_simulation_70B_modell
