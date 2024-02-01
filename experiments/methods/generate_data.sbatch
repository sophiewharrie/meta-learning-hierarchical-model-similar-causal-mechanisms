#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=4G
#SBATCH --output=experiments/output/slurm/output_syndata%A.txt

outprefix=$1
seed=$2
N_train=$3
N_val=$4
N_test=$5
M_train=$6
M_test=$7
C=$8
sigma_ref=$9
sigma_group=${10}
sigma_task=${11}
sigma_noise=${12}
eta_group=${13}
eta_task=${14}
p_interv=${15}

# create the data
mlflow run . -e create_synthetic_dataset --env-manager=local --experiment-name=synthetic_dataset_quality -P outprefix=${outprefix} -P seed=${seed} -P N_train=${N_train} -P N_val=${N_val} -P N_test=${N_test} -P M_train=${M_train} -P M_test=${M_test} -P C=${C} -P sigma_ref=${sigma_ref} -P sigma_group=${sigma_group} -P sigma_task=${sigma_task} -P sigma_noise=${sigma_noise} -P eta_group=${eta_group} -P eta_task=${eta_task} -P p_interv=${p_interv}
