# Running the experiment

1. Environment setup:

```
source experiment_env/bin/activate
module load anaconda
ulimit -c 0
```

1. Run the main experiment:

```
./experiments/3_real_data_experiments/experiment.sh
```

1. Analyse the results and make plots for publication:

```
analysis.ipynb
```
