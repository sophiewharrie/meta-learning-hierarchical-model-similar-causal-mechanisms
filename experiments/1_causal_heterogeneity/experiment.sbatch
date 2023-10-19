#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=4G
#SBATCH --output=experiments/output/slurm/output_%A.txt

datafile=$1
simfile=$2
metafile=$3
identifier=$4
num_groups=$5

experiment_name=synthetic_experiments
outprefix=experiments/output/synthetic/${identifier}

# global bnn baseline
for learning_rate in 0.001 0.0001
do
    outpath=${outprefix}_global_${learning_rate}
    sbatch experiments/methods/global_bnn_baseline.sbatch ${experiment_name} ${datafile} ${outpath} ${learning_rate}
done

# individual bnn baseline
for learning_rate in 0.001 0.0001
do
    outpath=${outprefix}_individual_${learning_rate}
    sbatch experiments/methods/individual_bnn_baseline.sbatch ${experiment_name} ${datafile} ${outpath} ${learning_rate}
done

# meta learning baseline
for base_learning_rate in 0.001 0.0001
do
    for meta_learning_rate in 0.01 0.001
    do
        outpath=${outprefix}_metalearner_${base_learning_rate}_${meta_learning_rate}
        sbatch experiments/methods/bayesian_metalearner_baseline.sbatch ${experiment_name} ${datafile} ${outpath} ${base_learning_rate} ${meta_learning_rate}
    done
done

# our method - known causal distances
for causal_distance in ground_truth SID SHD OD ID
do
    for base_learning_rate_initial in 0.001 0.0001
    do
        for meta_learning_rate_initial in 0.01 0.001
        do
            for base_learning_rate_main in 0.001 0.0001
            do
                for meta_learning_rate_global_main in 0.001 0.0001
                do
                    for meta_learning_rate_group_main in 0.001 0.0001
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
    for base_learning_rate_initial in 0.001 0.0001
    do
        for meta_learning_rate_initial in 0.01 0.001
        do
            for base_learning_rate_main in 0.001 0.0001
            do
                for meta_learning_rate_global_main in 0.001 0.0001
                do
                    for meta_learning_rate_group_main in 0.001 0.0001
                    do
                        num_initial_epochs=20
                        lambda=0.01
                        interventional_option=4
                        outpath=${outprefix}_ourmethodunknownCGM_${inference_type}_${interventional_option}_${base_learning_rate_initial}_${meta_learning_rate_initial}_${base_learning_rate_main}_${meta_learning_rate_global_main}_${meta_learning_rate_group_main}
                        sbatch experiments/methods/our_method_unknown_CGMs.sbatch ${experiment_name} ${datafile} ${num_groups} ${inference_type} ${outpath} ${base_learning_rate_initial} ${meta_learning_rate_initial} ${base_learning_rate_main} ${meta_learning_rate_global_main} ${meta_learning_rate_group_main} ${num_initial_epochs} ${lambda} ${interventional_option}
                    done
                done
            done
        done
    done
done
