#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --output=experiments/output/slurm/output_metalearner%A.txt

mlflow run . -e bayesian_metalearner_model --env-manager=local --experiment-name=$1 -P datafile=$2 -P outprefix=$3 -P base_learning_rate=$4 -P meta_learning_rate=$5
