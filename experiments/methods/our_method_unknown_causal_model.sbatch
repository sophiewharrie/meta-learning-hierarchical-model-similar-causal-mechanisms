#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --mem=8G
#SBATCH --output=experiments/output/slurm/output_oursunknown%A.txt

ulimit -c 0 # prevent core dump

mlflow run . -e our_method_unknown_causal_structure --env-manager=local --experiment-name=$1 -P datafile=$2 -P intervfile=$3 -P causal_distance=inferred -P num_groups=$4 -P outprefix=$5 -P num_pretrain_epochs=$6 -P num_initial_epochs=$7 -P num_main_epochs=$8 -P meta_learning_rate_global=$9 -P meta_learning_rate_group=${10} -P base_learning_rate=${11} -P alpha=${12} -P lambda=${13} -P causal_prior=${14} -P causal_model=${15} -P bandwidth_z=${16} -P bandwidth_theta=${17}