#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=4G
#SBATCH --output=experiments/output/slurm/output_%A.txt

datafile=$1
identifier=$2
experiment_name=$3

outprefix=experiments/output/real/${identifier}

# ablation study for interventional distance
for inference_type in interventional
do
    for interventional_option in 2 0
    do
        for num_groups in 4
        do
            for lambda in 0.001
            do
                for num_initial_epochs in 5
                do
                    for base_learning_rate_initial in 0.0001
                    do
                        for meta_learning_rate_initial in 0.001
                        do
                            for base_learning_rate_main in 0.001
                            do
                                for meta_learning_rate_global_main in 0.01
                                do
                                    for meta_learning_rate_group_main in 0.001
                                    do
                                        outpath=${outprefix}_ourmethodunknownCGM_${inference_type}_${interventional_option}_${num_groups}_${num_initial_epochs}_${lambda}_${base_learning_rate_initial}_${meta_learning_rate_initial}_${base_learning_rate_main}_${meta_learning_rate_global_main}_${meta_learning_rate_group_main}
                                        sbatch experiments/methods/our_method_unknown_CGMs.sbatch ${experiment_name} ${datafile} ${num_groups} ${inference_type} ${outpath} ${base_learning_rate_initial} ${meta_learning_rate_initial} ${base_learning_rate_main} ${meta_learning_rate_global_main} ${meta_learning_rate_group_main} ${num_initial_epochs} ${lambda} ${interventional_option}
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
