#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=4G
#SBATCH --output=experiments/output/slurm/output_syndata%A.txt

outprefix=$1
ref_input=$2
seed=$3
N_train=$4
N_val=$5
N_test=$6
M_train=$7
M_test=$8
C=$9
sigma_ref=${10}
sigma_group=${11}
sigma_task=${12}
sigma_noise=${13}
eta_group=${14}
eta_task=${15}

# create the data
mlflow run . -e create_synthetic_dataset --env-manager=local --experiment-name=synthetic_dataset_quality -P outprefix=${outprefix} -P ref_input=${ref_input} -P seed=${seed} -P N_train=${N_train} -P N_val=${N_val} -P N_test=${N_test} -P M_train=${M_train} -P M_test=${M_test} -P C=${C} -P sigma_ref=${sigma_ref} -P sigma_group=${sigma_group} -P sigma_task=${sigma_task} -P sigma_noise=${sigma_noise} -P eta_group=${eta_group} -P eta_task=${eta_task}
