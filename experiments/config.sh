tabular_datafile="path/to/mainfile.csv"
metafile="path/to/metafile.csv"
longitudinal_datafile="path/to/longfile.csv"
distancefile_BASELINE="path/to/distfile_BASELINE_method.csv"
distancefile_MR="path/to/distfileMR_method.csv"
distancefile_ICP="path/to/distfile_ICP_method.csv"
distancefile_SCM="path/to/distfile_SCM_method.csv"
outprefix="path/to/outdir/prefix"
learning_type=transductive
test_frac=0.5
query_frac=0.5
case_frac=0.5
batch_size=200
random_seed=42
n_kfold_splits=2
minibatch_size=5
max_num_epochs_baseline=10
max_num_epochs=5
num_mc_samples=10
wandb_n_trials=3
wandb_eval=AUCROC
wandb_direction=maximize
wandb_key_file=credentials/wandb_api_key.txt
wandb_project_prefix=test-run
wandb_entity=wandb-entity
mode=regular