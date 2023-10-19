#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=4G
#SBATCH --output=experiments/output/slurm/output_%A.txt

datafile=$1
input_simfile=$2
simfile=$3
metafile=$4
identifier=$5
num_groups=$6

experiment_name=alpha_experiment
outprefix=experiments/output/synthetic/${identifier}

# add the causal distances to the dataset
python3 ./experiments/2_alpha_analysis/causal_dist.py ${input_simfile} ${simfile}

# our method - known causal distances
for causal_distance in TOD_struc1 TOD_struc2 TOD_struc3 TOD_struc4 TOD_struc5 TOD_struc6 TID_struc1 TID_struc2 TID_struc3 TID_struc4 TID_struc5 TID_struc6 TOD_dist1 TOD_dist2 TOD_dist3 TOD_dist4 TOD_dist5 TOD_dist6 TID_dist1 TID_dist2 TID_dist3 TID_dist4 TID_dist5 TID_dist6
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