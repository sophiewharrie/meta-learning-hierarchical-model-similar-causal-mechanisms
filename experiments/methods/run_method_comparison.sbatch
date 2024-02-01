#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=4G
#SBATCH --output=experiments/output/slurm/output_%A.txt

# a utility script to run a method comparison with 
# different hyperparameter configurations 
# for a given dataset
# and log all the results

# give as input the paths to the datafiles 
datafile=$1
metafile=$2 # NOTE: for real data, give metafile='ignore', which should be ignored when causal_distance!=ground_truth
intervfile=$3
skip_known=$4 # if yes skip the method for known causal models, else if no don't skip this method

experiment_name=$5 # name for the experiment group
identifier=$6 # identifier for this particular experiment
num_groups=$7 # number of groups to use for methods that need this
outprefix=experiments/output/data/${identifier}

# large global bnn baseline (2x number of parameters)
for learning_rate in 0.001 0.0001
do
    outpath=${outprefix}_globallarge_${learning_rate}
    sbatch experiments/methods/global_bnn_baseline_large.sbatch ${experiment_name} ${datafile} ${outpath} ${learning_rate}
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
    for meta_learning_rate in 0.001 0.0001
    do
        outpath=${outprefix}_metalearner_${base_learning_rate}_${meta_learning_rate}
        sbatch experiments/methods/bayesian_metalearner_baseline.sbatch ${experiment_name} ${datafile} ${outpath} ${base_learning_rate} ${meta_learning_rate}
    done
done

# TSA-MAML baseline
for base_learning_rate in 0.001 0.0001
do
    for meta_learning_rate in 0.001 0.0001
    do
        for C in ${num_groups}
        do
            outpath=${outprefix}_tsamaml_${base_learning_rate}_${meta_learning_rate}
            sbatch experiments/methods/TSAMAML.sbatch ${experiment_name} ${datafile} ${outpath} ${C} ${meta_learning_rate} ${base_learning_rate} ${meta_learning_rate} ${base_learning_rate} 
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

# our method - known causal models (mostly used for testing purposes)
if [ "$skip_known" = "no" ]; then
    for num_pretrain_epochs in 200
    do
        for num_initial_epochs in 10
        do
            for num_main_epochs in 100
            do
                for meta_learning_rate_global in 0.001 0.0001
                do
                    for meta_learning_rate_group in 0.001 0.0001
                    do
                        for base_learning_rate in 0.001 0.0001
                        do
                            for alpha in 0.01 
                            do 
                                for lambda in 0.01
                                do  
                                    for C in ${num_groups}
                                    do
                                        causal_prior=erdos_renyi
                                        outpath=${outprefix}_ourmethodknowncausalmodels_${C}_${num_pretrain_epochs}_${num_initial_epochs}_${num_main_epochs}_${meta_learning_rate_global}_${meta_learning_rate_group}_${base_learning_rate}_${alpha}_${lambda}
                                        sbatch experiments/methods/our_method_known_causal_model.sbatch ${experiment_name} ${datafile} ${intervfile} ${metafile} ${C} ${outpath} ${num_pretrain_epochs} ${num_initial_epochs} ${num_main_epochs} ${meta_learning_rate_global} ${meta_learning_rate_group} ${base_learning_rate} ${alpha} ${lambda} ${causal_prior}
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done 
    done
else
    echo "Not using method for known causal models"
fi

# our method - unknown causal models
for num_pretrain_epochs in 500
do
    for num_initial_epochs in 10
    do
        for num_main_epochs in 100
        do
            for meta_learning_rate_global in 0.001 0.0001
            do
                for meta_learning_rate_group in 0.001 0.0001
                do
                    for base_learning_rate in 0.001 0.0001
                    do
                        for causal_model in linear non_linear
                        do 
                            for causal_prior in erdos_renyi scale_free
                            do
                                for lambda in 0.01
                                do
                                    for C in ${num_groups}
                                    do  
                                        # parameter config 1
                                        alpha=0.02
                                        bandwidth_z=5
                                        bandwidth_theta=1000
                                        outpath=${outprefix}_ourmethodunknowncausalmodels_${C}_${num_pretrain_epochs}_${num_initial_epochs}_${num_main_epochs}_${meta_learning_rate_global}_${meta_learning_rate_group}_${base_learning_rate}_${causal_model}_${causal_prior}_${lambda}_${alpha}_${bandwidth_z}_${bandwidth_theta}
                                        sbatch experiments/methods/our_method_unknown_causal_model.sbatch ${experiment_name} ${datafile} ${intervfile} ${C} ${outpath} ${num_pretrain_epochs} ${num_initial_epochs} ${num_main_epochs} ${meta_learning_rate_global} ${meta_learning_rate_group} ${base_learning_rate} ${alpha} ${lambda} ${causal_prior} ${causal_model} ${bandwidth_z} ${bandwidth_theta}
                                        # parameter config 2
                                        alpha=0.015
                                        bandwidth_z=8
                                        bandwidth_theta=1200
                                        outpath=${outprefix}_ourmethodunknowncausalmodels_${C}_${num_pretrain_epochs}_${num_initial_epochs}_${num_main_epochs}_${meta_learning_rate_global}_${meta_learning_rate_group}_${base_learning_rate}_${causal_model}_${causal_prior}_${lambda}_${alpha}_${bandwidth_z}_${bandwidth_theta}
                                        sbatch experiments/methods/our_method_unknown_causal_model.sbatch ${experiment_name} ${datafile} ${intervfile} ${C} ${outpath} ${num_pretrain_epochs} ${num_initial_epochs} ${num_main_epochs} ${meta_learning_rate_global} ${meta_learning_rate_group} ${base_learning_rate} ${alpha} ${lambda} ${causal_prior} ${causal_model} ${bandwidth_z} ${bandwidth_theta}
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
