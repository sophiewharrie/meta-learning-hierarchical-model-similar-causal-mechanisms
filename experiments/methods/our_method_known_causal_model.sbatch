#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --mem=8G
#SBATCH --output=experiments/output/slurm/output_oursknown%A.txt

ulimit -c 0 # prevent core dump

mlflow run . -e our_method_known_causal_structure --env-manager=local --experiment-name=$1 -P datafile=$2 -P intervfile=$3 -P metafile=$4 -P causal_distance=ground_truth -P num_groups=$5 -P outprefix=$6 -P num_pretrain_epochs=$7 -P num_initial_epochs=$8 -P num_main_epochs=$9 -P meta_learning_rate_global=${10} -P meta_learning_rate_group=${11} -P base_learning_rate=${12} -P alpha=${13} -P lambda=${14} -P causal_prior=${15}