# SCM method (DAGs)

## Setup:

Download the modified DiBS package code (and its dependencies) from https://github.com/sophiewharrie/dibs

```
git clone https://github.com/sophiewharrie/dibs
```

Organize the directory structure so that the `dibs` directory is in the same directory as the `causality-pheno-pred` directory, i.e.

```
root
    |_ causality-pheno-pred
    |_ dibs
```

## Usage:

1. Activate the Python environment for this repository (and make sure added DiBS dependencies are also installed)

2. The code uses JAX. If there's no GPU available on your machine, first run the following in the terminal:

```
export JAX_PLATFORMS=cpu
```

3. Run the Python code in `causal_distances/SCM/scm.py`, giving the prefix for the output as a command line argument `outprefix`, e.g., 

```
python3 ./causal_distances/SCM/scm.py --outprefix /path/to/outdir/prefix
```

this will save the output in the file `{outprefix}_causal_distance_SCM_method.csv`

[IF YOU HAVE A HPC CLUSTER] Run the following to launch a job on GPU for this script:

```
sbatch causal_distances/SCM/run_scm.sbatch
```


## Notes on usage:

- There are additional hyperparameters with values hard-coded in the `scm.py` file. Update these according to your needs (see the DiBS package documentation for guidance on this)
- Additionally, an interventional mask file can be provided. This is not a requirement of the code but can aid identifiability of causal relationships if information about hard interventions is available for your data. This is a binary mask of the same shape as the main data (matrix of shape `[N,d]`, where `N` is the number of samples and `d` is the number of vertices in the graph) that indicates whether or not a variable was intervened upon in a given sample