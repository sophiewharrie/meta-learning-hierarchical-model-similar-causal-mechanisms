# Meta-learning with hierarchical models based on similarity of causal mechanisms

This repository contains the implementation of the machine learning model and experiments as described in the research paper `Meta-learning with hierarchical models based on similarity of causal mechanisms`.

## Directory Structure

Below is an overview of the main components and layout of this repository:

```
├── README.md               # Overview and instructions related to the repository
├── LICENSE                 # Software license
├── MLproject               # MLflow project configuration
├── requirements.txt        # Python package dependencies
│
├── baselines/              # Implementations of baseline models
│
├── data/                   # Datasets and preprocessing scripts
│
├── data_generator/         # Scripts for generating synthetic data
│
├── experiments/            # Experiment configurations and result logs
│
├── metrics/                # Custom metrics used for model evaluation
│
├── model/                  # Main model implementation scripts
│
└── utils/                  # Utility scripts and helper functions
```


## Requirements and Setup

### Python Virtual Environment

To isolate the dependencies for this project, it's recommended to use a virtual environment. Here's how to set it up:

#### Create a Virtual Environment

```
python3 -m venv experiment_env
```

#### Activate the Virtual Environment

```
source experiment_env/bin/activate
```

#### Install the Dependencies

```
pip install -r requirements.txt
```

### Dependencies

Please refer to `requirements.txt` to recreate the environment used in experiments.

You will also need to clone this fork of the DiBS package: TODO.

## Usage

These instructions provide an example of how to run our method for Meta-learning with hierarchical models based on similarity of causal mechanisms. The code implementation uses a Bayesian Neural Network (BNN) with 2 Bayesian linear layers for regression.

### Input Data Format

#### Generating a Synthetic Dataset

1. Make sure you have activated the virtual environment

```
source experiment_env/bin/activate
```

2. Create a directory for the synthetic dataset

```
mkdir -p data/synthetic/output
```

3. Generate a synthetic dataset using the MLproject entry point for generating synthetic data

```
mlflow run . -e create_synthetic_dataset --env-manager=local --experiment-name=generate_synthetic_data
```

This uses the default hyperparameter settings, but please refer to the MLproject file to change the hyperparameter settings. If you would like a minimal example (e.g., just to test if the code works on your machine), then change N_train, N_val and N_test to low values so that the data generation and meta-learning methods run faster.

#### Using Your Own Dataset

If you are using your own dataset, prepare your data in the following format:

| X1 | X2 | ... | Y | task | task_train | meta_train |
|---|---|---|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

- Features (continuous or discrete values): column names beginning with `X`
- Label (continuous values): the label column should be called `Y`
- Task (ordinal values 0,1,...): each task is labelled with an index 0, 1, ..., in the `task` column
- Train/validation/test split (values 0, 1 or 2): the `task_train` column specifies if a task is in the train (0), validation (1) or test (2) set
- Support/query split (values 0 or 1): the `meta_train` column specifies which data samples are support (1) samples and query (0) samples

Also provide an intervention mask file, with the same columns as the dataset. The feature columns should have binary values indicating if the feature is an intervention target for the data sample.

If the causal models of the tasks are known, you can also also provide a file specifying the causal distances between each pair of tasks (otherwise, use the option for unknown causal models):

| task1 | task2 | causal_distance1 | causal_distance2 | ... |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

- Task pairs: index the task pairs using the task IDs from the `task` column of the main data file
- Causal distances: add a column for each causal distance metric you want to consider (SHD, etc.), indicating the causal distance between `task1` and `task2`

See the synthetic data generator for an example of the input formats.

### Running the Method

1. Make sure you have activated the virtual environment

```
source experiment_env/bin/activate
```

2. Create a directory for the results

```
mkdir -p experiments/output/test
```

3, Run our method using the MLproject entry point `our_method_unknown_causal_structure`:

```
mlflow run . -e our_method_unknown_causal_structure --env-manager=local --experiment-name=method_causal_unknown
```

This uses the default hyperparameter settings, but please refer to the MLproject file and the paper for further details about what other hyperparameter settings are available.

### Checking the Results

Our code uses [MLflow](https://mlflow.org/) for parameter and metric tracking. Each time a method is run, relevant parameters, metrics, and artifacts are logged to MLflow. This ensures a consistent and organized way to monitor, compare, and reproduce results.

You can find the logs from each method run in the `mlruns` directory. See https://mlflow.org/docs/latest/tracking.html for further instructions on how you can check the results using the tracking UI and/or terminal.

## Reproducing Experiments From the Paper

All experiments associated with the research paper can be found in the `experiments` directory. Each experiment has its own subdirectory,

## Datasets

All details for preparing the datasets used in experiments can be found in the `datasets` directory. Each dataset has its own subdirectory.


### Running Experiments on HPC using Slurm

To facilitate running our experiments on High Performance Computing (HPC) clusters, we provide Slurm scripts. Slurm is a job scheduler that enables the allocation of resources and execution of jobs on large clusters.

Please see the `README.md` file in each experiment directory for detailed instructions. We summarise the main steps below.

1. Activate the Python virtual environment for this repository

```
source experiment_env/bin/activate
```

2. Run the shell script (from the root directory of this repository) that submits the Slurm script

```
./experiments/{REPLACE-WITH-NAME-OF-EXPERIMENT-DIRECTORY}/experiment.sh
```

Please adapt the Slurm script parameters such as number of nodes, CPUs, and memory as per your HPC cluster specifications.

### MLflow Integration with Experiments

[MLflow](https://mlflow.org/) is an open-source platform to manage the end-to-end machine learning lifecycle. In our experiments, MLflow is integrated for parameter and metric tracking. Each time an experiment is run, relevant parameters, metrics, and artifacts are logged to MLflow. This ensures a consistent and organized way to monitor, compare, and reproduce results.

Refer to our scripts to see specific details on how parameters and metrics are logged.

### MLproject File

MLflow's MLproject file defines the environment and entry points for methods (and baselines), along with parameters they accept. This makes it straightforward to share and reproduce experiments. In our repository, each method used in experiments has its own entry point in the MLproject file. 

The Slurm scripts for the experiments handle the code for running methods from the MLproject file and specification of parameters. Please note that we do not provide the code for methods from other authors used in our experiments (due to licensing reasons), but the code from the original authors can be accessed at the following sources:

- HSML [link to repository](https://github.com/huaxiuyao/HSML)
- TSA-MAML [link to repository](https://github.com/Carbonaraa/TSA-MAML)

### Jupyter Notebooks for Results Review

After running an experiment, refer to the Jupyter Notebook file in the corresponding experiment directory to review the experiment results logged by MLflow.

Navigate through the notebook cells to visualize results, and generate the tables and figures as presented in the paper. The notebooks provide an interactive environment to manipulate and delve deep into the research findings.

## Acknowledgements

We acknowledge the following code packages and repositories that were especially useful for carrying out our research:
- [DiBS](https://github.com/larslorch/dibs)
- [Higher](https://github.com/facebookresearch/higher)
- [BLiTZ](https://github.com/piEsposito/blitz-bayesian-deep-learning/tree/master)
- [Causal Discovery Toolbox](https://github.com/FenTechSolutions/CausalDiscoveryToolbox)
- [NumPyro](https://github.com/pyro-ppl/numpyro) and [Pyro](https://github.com/pyro-ppl/pyro)
