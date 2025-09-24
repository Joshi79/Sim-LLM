# Sim-LLM: Enhancing RL with the use of LLM Environment Simulators 
Author: Joshua Joel Haas


## Prerequisites
1.) Python 3.12 \
2.) Install the required packages via pip install -r requirements.txt \
3.) Need to have access to Snelius \
4.) Download the LLaMa-3.1-8B-Instruct model from Hugging Face and transfer it to Snelius via Filesharing. \
5.) Need to have access to LLaMa-3.3.-70B Instruct via Servicedesk of SURF. \
6.) Need to transfer the orignal data to Snelius via Filesharing.

## Project Structure

This repository organizes code, data, and notebooks as follows:
- 'evaluation_notebooks': Contains evaluation scripts and notebooks.
- reports: Contains generated reports and visualizations.
- ''


## Model Access:
### Llama-3.3.-70B Instruct
To access the LLaMa-3.3.-70B Instruct you need to send a ticket to  https://servicedesk.surf.nl, with a screenshot of the model provider to access the model. 

### Llama-3.1-8B-Instruct
To access the LLaMa-3.1-8B-Instruct you need to download the model from Hugging Face: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct. Subsequently, to downloading the model you need to transfer via Filesharing in your directory of Snelius. 


### File Sharing 
To transfer files from your local Computer to Snelius Windows user can use WinSCP (https://winscp.net/eng/download.php).



#### Optuna Hyperparameter Tuning
The script for Optuna is made to run on Snelius and not locally. 
To run the optuna hyperparameter tuning you need to transfer the training data to your Snelius Project space 
and change the path in the code to the project space, in which the data is stored 


