#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --output=experiments/output/slurm/output_oursknown%A.txt

mlflow run . -e our_method_known_causal_structure --env-manager=local --experiment-name=$1 -P datafile=$2 -P simfile=$3 -P metafile=$4 -P num_groups=$5 -P causal_distance=$6 -P outprefix=$7 -P base_learning_rate_initial=$8 -P meta_learning_rate_initial=$9 -P base_learning_rate_main=${10} -P meta_learning_rate_global_main=${11} -P meta_learning_rate_group_main=${12} -P lambda=${13}