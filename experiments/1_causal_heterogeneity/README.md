# Experiment: Causal Heterogeneity Analysis

## Description

In this synthetic data experiment we compare our method with the global, local and Bayesian meta-learning baselines. For our method we study two settings of causal distances for known CGMs and proxies for unknown CGMs. In both cases the ground truth causal groups are unknown. The experiment reports the RMSEs for the test tasks, for various values of $C$, and the F1-scores for the recovery of ground truth causal group assignments.

## Instructions for Running the Experiment

### Preliminaries and Setup

Preliminaries:
- This experiment uses synthetic data generated in the synthetic data validation experiment. See `experiments/0_synthetic_data_validation/README.md` for further details

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
./experiments/1_causal_heterogeneity/experiment.sh
```

### Reviewing Results with Jupyter Notebook

To view detailed results and analysis, use the provided Jupyter notebook: `analysis.ipynb`. Navigate through the notebook cells to visualize results, and generate the tables and figures as presented in the paper.