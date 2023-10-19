#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --output=experiments/output/slurm/output_hsml%A.txt

mlflow run . -e hsml_model --env-manager=conda --experiment-name=$1 -P datafile=$2 -P outdir=$3 -P update_lr=$4 -P meta_lr=$5