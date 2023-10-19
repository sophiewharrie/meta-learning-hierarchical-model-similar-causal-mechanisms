# Experiment: Generalisation Analysis

## Description

For standard meta-learning methods, environment-level and task-level generalisation gaps result from a finite number of tasks and number of samples per task. This experiment analyses these effects on generalisation performance for our method (fixed $C=4$).

## Instructions for Running the Experiment

### Preliminaries and Setup

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
./experiments/3_N_m_analysis/experiment.sh
```

### Reviewing Results with Jupyter Notebook

To view detailed results and analysis, use the provided Jupyter notebook: `analysis.ipynb`. Navigate through the notebook cells to visualize results, and generate the tables and figures as presented in the paper.
