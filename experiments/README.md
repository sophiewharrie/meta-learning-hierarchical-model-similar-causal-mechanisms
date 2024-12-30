Experiments for Stroke prediction case study (supervised learning, binary classification tasks, with longitudinal data)

Usage:
1. **Check config file:** The `config.sh` file provides all the settings used in each experiment. Update this file before launching an experiment run. In particular, check `outprefix` and `wandb_project_prefix`, which should be unique for each experiment. Make sure that the directory for `outprefix` exists
2. **Check sbatch files:** Check that the `.sbatch` files are configured to provide adequate computing resources for your data. In particular, you can increase the number of array jobs to speed up hyperparameter tuning
3. **Check experiment file:** The `experiment.sh` script describes which experiments to run (which ML methods, causal methods, etc.). Edit this file if you want to run a different set of experiments

**Launching the experiments:**

From the root directory of this repository:

4. Activate the Python environment for this repository (see main README for dependencies)
5. Make a slurm directory for the output (if it doesn't already exist): `mkdir slurm` 
6. Launch the experiments by running the `experiments.sh` script:

```
# from the root directory of this repository:
chmod +x experiments/experiment.sh
./experiments/experiment.sh
```

This will create a Weights and Biases project for each experiment, where you can check the hyperparameter tuning results.


## Troubleshooting

- If the jobs don't launch properly, check there isn't already a sweep ID file in the `outprefix` directory