#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --output=experiments/output/slurm/output_individual%A.txt

mlflow run . -e individual_bnn_model --env-manager=local --experiment-name=$1 -P datafile=$2 -P outprefix=$3 -P learning_rate=$4