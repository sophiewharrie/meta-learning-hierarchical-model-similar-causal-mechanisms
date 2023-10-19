#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --output=experiments/output/slurm/output_oursunknown%A.txt

mlflow run . -e our_method_unknown_causal_structure --env-manager=local --experiment-name=$1 -P datafile=$2 -P num_groups=$3 -P inference_type=$4 -P outprefix=$5 -P base_learning_rate_initial=$6 -P meta_learning_rate_initial=$7 -P base_learning_rate_main=$8 -P meta_learning_rate_global_main=${9} -P meta_learning_rate_group_main=${10} -P num_initial_epochs=${11} -P lambda=${12} -P interventional_option=${13}