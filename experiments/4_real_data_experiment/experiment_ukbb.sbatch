#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=4G
#SBATCH --output=experiments/output/slurm/output_%A.txt

datafile=$1
identifier=$2
experiment_name=$3

outprefix=experiments/output/real/${identifier}

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

# our method - unknown causal distances 
for inference_type in interventional observational
do
    for interventional_option in 4
    do
        for num_groups in 2 4 6
        do
            for lambda in 0.01 0.001
            do
                for num_initial_epochs in 5 10
                do
                    for base_learning_rate_initial in 0.001 0.0001
                    do
                        for meta_learning_rate_initial in 0.01 0.001
                        do
                            for base_learning_rate_main in 0.001
                            do
                                for meta_learning_rate_global_main in 0.001 0.0001
                                do
                                    for meta_learning_rate_group_main in 0.001 0.0001
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

# TSA-MAML baseline
for base_learning_rate in 0.001 0.0001
do
    for meta_learning_rate in 0.001 0.0001
    do
        for num_groups in 2 4 6
        do
            outpath=${outprefix}_tsamaml_${base_learning_rate}_${meta_learning_rate}
            sbatch experiments/methods/TSAMAML.sbatch ${experiment_name} ${datafile} ${outpath} ${num_groups} ${meta_learning_rate} ${base_learning_rate} ${meta_learning_rate} ${base_learning_rate} 
        done
    done
done

# HSML baseline
for base_learning_rate in 0.001 0.0001
do
    for meta_learning_rate in 0.001 0.0001
    do
        outdir=baselines/HSML/logs/${identifier}_${base_learning_rate}_${meta_learning_rate}/
        sbatch experiments/methods/HSML.sbatch ${experiment_name} ${datafile} ${outdir} ${base_learning_rate} ${meta_learning_rate}
    done
done

# ARML baseline
for base_learning_rate in 0.001 0.0001
do
    for meta_learning_rate in 0.001 0.0001
    do
        outdir=baselines/ARML/logs/${identifier}_${base_learning_rate}_${meta_learning_rate}/
        sbatch experiments/methods/ARML.sbatch ${experiment_name} ${datafile} ${outdir} ${base_learning_rate} ${meta_learning_rate}
    done
done