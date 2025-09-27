# Sim-LLM: Enhancing RL with the use of LLM Environment Simulators 
Author: Joshua Joel Haas



## Project structure

```text
Sim-LLM/
├── data/                  # Data files (ignored by git)
├── evaluation_notebooks/  # Jupyter notebooks for evaluation and analysis
├── reports/                # Generated reports, plots, and trained DDQNs
├── slurm_jobs/              # SLURM scripts for running jobs on Snelius
├── src/                   # Source code
│   ├── data_preperation/  # Scripts for data preprocessing and splitting
│   ├── offline_evaluation/ # Scripts for offline evaluation of DDQN 
│   ├── simulation/        # Scripts for running the LLM-based simulations
│   └── visualisations/    # Scripts for generating visualisations
├── .gitignore
├── README.md
└── requirements.txt
```
## Prerequisites
```text
1.) Python 3.12 
2.) Install the required packages via pip install -r requirements.txt 
3.) Need to have access to Snelius 
4.) Download the LLaMa-3.1-8B-Instruct model from Hugging Face and transfer it to Snelius via Filesharing. 
5.) Need to have access to LLaMa-3.3.-70B Instruct via Servicedesk of SURF. 
6.) Need to transfer the orignal data to Snelius via Filesharing.
```


## Configuration

### Model Access:

### Llama-3.3.-70B Instruct
To access the LLaMa-3.3.-70B Instruct you need to send a ticket to  https://servicedesk.surf.nl, with a screenshot of the model provider to access the model. 

### Llama-3.1-8B-Instruct
To access the LLaMa-3.1-8B-Instruct you need to download the model from Hugging Face: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct. Subsequently, to downloading the model you need to transfer via Filesharing in your directory of Snelius. 

### File Sharing 
To transfer files from your local Computer to Snelius Windows user can use WinSCP (https://winscp.net/eng/download.php).

### Usage 

##### Optuna Hyperparameter Tuning
To run the Optuna hyperparameter tuning for the DDQN agent, submit the ex_optuna.sh job:
Before running the Job make sure to change the path to your data path on snelius before running the job

```bash
sbatch ex_optuna.sh
```

##### Simulation 70B
To run the simulation with the 70B model, you need to change the path to your data path on Snelius on the prompt_parts.py file
Subsequently, you can change the number of patients and number of days in the run_simulations_70B_modell.py
Also you need the to request access to the LLaMa-3.3.-70B Instruct model via Servicedesk of SURF.
To run the simulation, submit the ex_simulation_70B.sh job:

```bash
sbatch ex_llama70B.sh
```

##### Simulation 8B
To run the simulation with the 70B model, you need to change the path to your data path on Snelius on the prompt_parts.py file
You need to download the LLaMa-3.1-8B-Instruct model from Hugging Face and transfer it to Snelius via Filesharing.
Subsequently, you need to change the data directory in which you saved the model in the llama_3_1_8b_instruct.py file
To run the simulation, submit the ex_simulation_8B.sh job:

```bash
sbatch ex_llama_8B.sh
```