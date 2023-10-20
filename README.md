# Causal Similarity-Based Hierarchical Bayesian Models

This repository contains the implementation of the machine learning model and experiments as described in the research paper [Causal Similarity-Based Hierarchical Bayesian Models](https://arxiv.org/abs/2310.12595).

## Citation

If you find this code useful in your research, please consider citing:

```
@article{wharrie2023causal,
  title={Causal Similarity-Based Hierarchical Bayesian Models},
  author={Wharrie, Sophie and Kaski, Samuel},
  journal={arXiv preprint arXiv:2310.12595},
  year={2023},
}
```

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

The primary dependencies include:
- blitz-bayesian-pytorch==0.2.8
- cdt==0.6.0
- higher==0.2.1
- jax==0.4.11
- matplotlib==3.8.0
- mlflow==2.3.2
- networkx==3.1
- numpy==1.24.4
- numpyro==0.12.1
- pandas==2.1.1
- pyemd==1.0.0
- scikit-learn==1.3.1
- scipy==1.11.3
- torch==2.0.1

Please refer to `requirements.txt` for a complete list.

## Usage

These instructions provide an example of how to run our method for Causal Similarity-Based Hierarchical Bayesian Models. The code implementation uses a Bayesian Neural Network (BNN) with 2 Bayesian linear layers for regression.

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
mlflow run . -e create_synthetic_dataset --env-manager=local --experiment-name=generate_synthetic_data -P outprefix=data/synthetic/output/test -P ref_input=data_generator/example_edgefile.txt -P seed=1234 -P N_train=20 -P N_val=5 -P N_test=10 -P M_train=5 -P M_test=5 -P C=2 -P sigma_ref=1.0 -P sigma_group=0.1 -P sigma_task=0.0001 -P sigma_noise=0.1 -P eta_group=0.6 -P eta_task=0.05
```

This creates a synthetic dataset with 20 train tasks, 5 validation tasks and 10 test tasks, for 2 groups of causally similar tasks. Each task has 5 support samples and 5 query samples. Please refer to the MLproject file and the paper for further details about what the hyperparameters represent.

#### Using Your Own Dataset

If you are using your own dataset, prepare your data in the following format:

| X1 | X2 | ... | T1 | T2 | ... | Y | task | task_train | meta_train |
|---|---|---|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

- Features (continuous or discrete values): the feature list consists of covariates variables (column names beginning with `X`) and intervention variables (column names beginning with `T`). Please note that the code has only been tested with binary intervention variables
- Label (continuous values): the label column should be called `Y`
- Task (ordinal values 0,1,...): each task is labelled with an index 0, 1, ..., in the `task` column
- Train/validation/test split (values 0, 1 or 2): the `task_train` column specifies if a task is in the train (0), validation (1) or test (2) set
- Support/query split (values 0 or 1): the `meta_train` column specifies which data samples are support (1) samples and query (0) samples

If the causal models of the tasks are known, you can also also provide a file specifying the causal distances between each pair of tasks (otherwise, use the option for unknown causal models):

| task1 | task2 | causal_distance1 | causal_distance2 | ... |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

- Task pairs: index the task pairs using the task IDs from the `task` column of the main data file
- Causal distances: add a column for each causal distance metric you want to consider (SHD, SID, OD, ID, etc.), indicating the causal distance between `task1` and `task2`

### Running the Method

1. Make sure you have activated the virtual environment

```
source experiment_env/bin/activate
```

2. Create a directory for the results

```
mkdir -p experiments/output/test
```

#### Known Causal Models

Run our method for known causal models using the MLproject entry point `our_method_known_causal_structure`:

```
mlflow run . -e our_method_known_causal_structure --env-manager=local --experiment-name=method_causal_known -P datafile=data/synthetic/output/test_data.csv -P simfile=data/synthetic/output/test_causal_sim.csv -P num_groups=2 -P causal_distance=SHD -P outprefix=experiments/output/test/synthetic_known
```

In this example we use the Structural Hamming Distance (SHD) as the (known) causal distance. We have left most other hyperparameter settings on the default values, but please refer to the MLproject file and the paper for further details about what other hyperparameter settings are available.


#### Unknown Causal Models

Run our method for unknown causal models using the MLproject entry point `our_method_unknown_causal_structure`:

```
mlflow run . -e our_method_unknown_causal_structure --env-manager=local --experiment-name=method_causal_unknown -P datafile=data/synthetic/output/test_data.csv -P num_groups=2 -P inference_type=interventional -P outprefix=experiments/output/test/synthetic_unknown
```

In this example we use the Interventional Proxy (OP) as the proxy distance. We have left most other hyperparameter settings on the default values, but please refer to the MLproject file and the paper for further details about what other hyperparameter settings are available.

### Checking the Results

Our code uses [MLflow](https://mlflow.org/) for parameter and metric tracking. Each time a method is run, relevant parameters, metrics, and artifacts are logged to MLflow. This ensures a consistent and organized way to monitor, compare, and reproduce results.

You can find the logs from each method run in the `mlruns` directory. See https://mlflow.org/docs/latest/tracking.html for further instructions on how you can check the results using the tracking UI and/or terminal.

## Reproducing Experiments From the Paper

All experiments associated with the research paper can be found in the `experiments` directory. Each individual experiment has its own subdirectory, as listed below.

### Experiment List

- `experiments/0_synthetic_data_validation/`: validation of the synthetic data quality and causal properties
- `experiments/1_causal_heterogeneity/`: method comparison for synthetic datasets with varying degrees of causal heterogeneity and various causal distances
- `experiments/2_alpha_analysis/`: analysis of the effect of causal distance hyperparameters for synthetic datasets
- `experiments/3_N_m_analysis/`: analysis of the effect of N, M on generalisation for synthetic datasets
- `experiments/4_real_data_experiment/`: method comparison experiment for real datasets
- `experiments/5_interventional_ablation_study/`: ablation study of proxies for causal distances in real datasets
- `experiments/6_complexity_analysis/`: analysis of our method's computational complexity

### Datasets

The following datasets are used in experiments:

#### Synthetic Dataset

We provide code for generating synthetic datasets according to the toy model described in the paper. See the `data_generator` directory for further details.

#### Economics Dataset

The economics dataset uses the IMF World Economic Outlook dataset, April 2023 version ([link to download](https://www.imf.org/en/Publications/WEO/weo-database/2023/April)), and Penn World Table dataset, version 10.01 ([link to download](https://www.rug.nl/ggdc/productivity/pwt/?lang=en)). See the `data/economics` directory for the preprocessing code and `data/README.md` for more details on the preprocessing steps.

#### Medical Dataset

The medical dataset uses the UK Biobank data resource. This dataset is not publicly available for download, but researchers can register with UK Biobank and apply for data access ([link to more details](https://www.ukbiobank.ac.uk/)). See the `data/medical` directory for the preprocessing code and `data/README.md` for more details on the preprocessing steps.


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

This study has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 101016775. This research has been conducted using the UK Biobank Resource under Application Number 77565.

We acknowledge the following code packages and repositories that were especially useful for carrying out our research:
- [NumPyro](https://github.com/pyro-ppl/numpyro) and [Pyro](https://github.com/pyro-ppl/pyro)
- [Higher](https://github.com/facebookresearch/higher)
- [BLiTZ](https://github.com/piEsposito/blitz-bayesian-deep-learning/tree/master)
- [PyEMD](https://github.com/wmayner/pyemd)
- [Causal Discovery Toolbox](https://github.com/FenTechSolutions/CausalDiscoveryToolbox)
