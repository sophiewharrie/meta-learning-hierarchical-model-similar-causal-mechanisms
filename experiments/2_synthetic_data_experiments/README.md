# Running the experiment

0. Preliminaries: this experiment uses the data generated in the previous experiment

NOTE: update the parameters in the experiment.sh file if any of the values change

1. Environment setup:

```
source experiment_env/bin/activate
module load anaconda
ulimit -c 0
```

1. Run the experiment:

```
./experiments/2_synthetic_data_experiments/experiment.sh
```

1. Analyse the results and make plots for publication:

```
analysis.ipynb
```
