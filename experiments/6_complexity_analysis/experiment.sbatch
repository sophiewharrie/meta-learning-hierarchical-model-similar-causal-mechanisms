#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=4G
#SBATCH --output=experiments/output/slurm/output_%A.txt

datafile=$1
simfile=$2
metafile=$3
identifier=$4
trial=$5

experiment_name=complexity_experiments
data_outprefix=data/synthetic/output/${identifier}
ref_input=data_generator/example_edgefile.txt
seed=1234
N_train=$7
N_val=10
N_test=10
M_train=$8
M_test=$8
C=$6
sigma_ref=1.0
sigma_group=0.1
sigma_task=0.0001
sigma_noise=0.1
eta_group=0.6
eta_task=0.05
num_groups=${C}

outprefix=experiments/output/synthetic/${identifier}

# create the data
mlflow run . -e create_synthetic_dataset --env-manager=local --experiment-name=synthetic_dataset_generalisation -P outprefix=${data_outprefix} -P ref_input=${ref_input} -P seed=${seed} -P N_train=${N_train} -P N_val=${N_val} -P N_test=${N_test} -P M_train=${M_train} -P M_test=${M_test} -P C=${C} -P sigma_ref=${sigma_ref} -P sigma_group=${sigma_group} -P sigma_task=${sigma_task} -P sigma_noise=${sigma_noise} -P eta_group=${eta_group} -P eta_task=${eta_task}

# our method - known causal distances
for causal_distance in SHD
do
    for base_learning_rate_initial in 0.0001
    do
        for meta_learning_rate_initial in 0.001
        do
            for base_learning_rate_main in 0.0001
            do
                for meta_learning_rate_global_main in 0.001
                do
                    for meta_learning_rate_group_main in 0.001
                    do
                        lambda=0.01
                        outpath=${outprefix}_ourmethodknownCGM_${causal_distance}_${base_learning_rate_initial}_${meta_learning_rate_initial}_${base_learning_rate_main}_${meta_learning_rate_global_main}_${meta_learning_rate_group_main}
                        sbatch experiments/methods/our_method_known_CGMs.sbatch ${experiment_name} ${datafile} ${simfile} ${metafile} ${num_groups} ${causal_distance} ${outpath} ${base_learning_rate_initial} ${meta_learning_rate_initial} ${base_learning_rate_main} ${meta_learning_rate_global_main} ${meta_learning_rate_group_main} ${lambda}
                    done
                done
            done
        done
    done
done

# our method - unknown causal distances 
for inference_type in interventional observational
do
    for base_learning_rate_initial in 0.0001
    do
        for meta_learning_rate_initial in 0.001
        do
            for base_learning_rate_main in 0.0001
            do
                for meta_learning_rate_global_main in 0.001
                do
                    for meta_learning_rate_group_main in 0.001
                    do
                        num_initial_epochs=20
                        lambda=0.01
                        interventional_option=4
                        outpath=${outprefix}_ourmethodunknownCGM_${inference_type}_${interventional_option}_${num_groups}_${num_initial_epochs}_${lambda}_${base_learning_rate_initial}_${meta_learning_rate_initial}_${base_learning_rate_main}_${meta_learning_rate_global_main}_${meta_learning_rate_group_main}
                        sbatch experiments/methods/our_method_unknown_CGMs.sbatch ${experiment_name} ${datafile} ${num_groups} ${inference_type} ${outpath} ${base_learning_rate_initial} ${meta_learning_rate_initial} ${base_learning_rate_main} ${meta_learning_rate_global_main} ${meta_learning_rate_group_main} ${num_initial_epochs} ${lambda} ${interventional_option}
                    done
                done
            done
        done
    done
done
