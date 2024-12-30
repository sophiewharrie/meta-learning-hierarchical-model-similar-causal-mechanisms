#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu
#SBATCH --partition=gpu-a100-80g
#SBATCH --output=slurm/ukb_long_two_level_%A_%a.out
#SBATCH --array=1-10 # Adjust based on how many parallel jobs you want

config="$1"
source $config

method=2_level_hierarchical
data_type=sequence
model_outprefix=${outprefix}_${SLURM_ARRAY_TASK_ID}_${method}
sweep_id_filename=${outprefix}_${method}_sweep_id.txt
wandb_project=${wandb_project_prefix}-${method}

# Create the sweep and get the sweep ID (only done by the first job in the array)
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    mlflow run . -e run_ml --env-manager=local \
    -P tabular_datafile=${tabular_datafile} \
    -P longitudinal_datafile=${longitudinal_datafile} \
    -P metafile=${metafile} \
    -P outprefix=${model_outprefix} \
    -P data_type=${data_type} \
    -P learning_type=${learning_type} \
    -P test_frac=${test_frac} \
    -P query_frac=${query_frac} \
    -P case_frac=${case_frac} \
    -P batch_size=${batch_size} \
    -P method=${method} \
    -P random_seed=${random_seed} \
    -P n_kfold_splits=${n_kfold_splits} \
    -P minibatch_size=${minibatch_size} \
    -P max_num_epochs=${max_num_epochs} \
    -P num_mc_samples=${num_mc_samples} \
    -P wandb_n_trials=${wandb_n_trials} \
    -P wandb_eval=${wandb_eval} \
    -P wandb_direction=${wandb_direction} \
    -P wandb_key_file=${wandb_key_file} \
    -P sweep_id_filename=${sweep_id_filename} \
    -P wandb_project=${wandb_project} \
    -P wandb_entity=${wandb_entity} \
    -P mode=init
fi

# Wait for the sweep ID file to be created
while [ ! -f ${sweep_id_filename} ]; do sleep 30; done

# Run the wandb agent with the sweep ID
mlflow run . -e run_ml --env-manager=local \
  -P tabular_datafile=${tabular_datafile} \
  -P longitudinal_datafile=${longitudinal_datafile} \
  -P metafile=${metafile} \
  -P outprefix=${model_outprefix} \
  -P data_type=${data_type} \
  -P learning_type=${learning_type} \
  -P test_frac=${test_frac} \
  -P query_frac=${query_frac} \
  -P case_frac=${case_frac} \
  -P batch_size=${batch_size} \
  -P method=${method} \
  -P random_seed=${random_seed} \
  -P n_kfold_splits=${n_kfold_splits} \
  -P minibatch_size=${minibatch_size} \
  -P max_num_epochs=${max_num_epochs} \
  -P num_mc_samples=${num_mc_samples} \
  -P wandb_n_trials=${wandb_n_trials} \
  -P wandb_eval=${wandb_eval} \
  -P wandb_direction=${wandb_direction} \
  -P wandb_key_file=${wandb_key_file} \
  -P sweep_id_filename=${sweep_id_filename} \
  -P wandb_project=${wandb_project} \
  -P wandb_entity=${wandb_entity} \
  -P mode=${mode}