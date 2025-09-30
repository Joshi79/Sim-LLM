# Sim-LLM: Enhancing RL with the use of LLM Environment Simulators 
Author: Joshua Joel Haas


## Abstract 
```text
Integrating  reinforcement learning (RL) into mobile health (mHealth) applications offers a path to personalise interventions and might reduce high patient dropout rates. 
However, direct clinical deployment poses significant ethical and safety challenges. 
Simulators provide a solution for safely training and evaluating RL policies. Therefore, this thesis examines how effectively a pre-trained Large Language Model (LLM)-based patient simulator, seeded with aggregated statistics from an mHealth study on low-mood interventions, improves the development of personalised RL policies compared with using only the original dataset.
A four-layer prompt instructed two LLMs, Llama-3.3-70B-Instruct and Llama-3.1-8B-Instruct, to generate 100 synthetic patient trajectories for low-mood interventions. 
The 8B model failed to produce valid outputs, thus restricting the analysis to the 70B model, whose data were assessed for fidelity and utility in RL training While the simulator adhered to boundary constraints and captured most original categories, achieving a CAT score of 0.880, it failed to replicate the overall statistical structure. 
The simulated patient trajectories exhibited divergent temporal correlations, and only two variables passed distributional tests against the original data.
 Despite this limited fidelity, policies trained on the complete synthetic data achieved the highest median FQE value (10.253) on the test data and outperformed the policy trained on the original data (p = $5.4944\times10^{-13}$). 
 However, this performance was not statistically superior to a fixed baseline policy (p = 0.159). 
 These results suggest that even with imperfect simulators, useful synthetic data can be developed for personalized RL policies, but substantial improvements are necessary before it can reliably enhance real patient data. 
```
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