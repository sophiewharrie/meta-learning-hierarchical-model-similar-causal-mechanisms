# Experiment: Ablation Study for Proxy Distances

## Description

In this experiment we conduct an ablation study of the proxies for causal distances, to understand the relative importance of the techniques used to approximate the interventional proxy.

## Instructions for Running the Experiment

### Preliminaries and Setup

Preliminaries:
- Update the `.sbatch` scripts with the optimal hyperparameters determined in the previous experiment (for real datasets)

Setup:
1. Check the `.sbatch` script/s and adapt the parameters such as number of nodes, CPUs, and memory as per your HPC cluster specifications
2. Check the hyperparameter choices in the `.sbatch` and/or `.sh` script/s and adapt according to your experiment setup

### Activate Virtual Environment

Before running the experiment, activate the virtual environment to ensure all dependencies are available:


```
source experiment_env/bin/activate  # On Windows, use experiment_env\Scripts\activate
```

### Launching the Experiment

Run the experiment using the following shell command:

```
./experiments/5_interventional_ablation_study/experiment.sh
```

### Reviewing Results with Jupyter Notebook

To view detailed results and analysis, use the provided Jupyter notebook: `analysis.ipynb`. Navigate through the notebook cells to visualize results, and generate the tables and figures as presented in the paper. 

Note: get the results for the OP and IP from the previous experiment.