#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --output=experiments/output/slurm/output_tsamaml%A.txt

mlflow run . -e tsa_maml_model --env-manager=local --experiment-name=$1 -P datafile=$2 -P outprefix=$3 -P num_groups=$4 -P meta_learning_rate_initial=$5 -P base_learning_rate_initial=$6 -P meta_learning_rate_main=$7 -P base_learning_rate_main=$8